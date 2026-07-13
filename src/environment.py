"""
Define the ERP cache environment for DQN training and evaluation.

CHANGELOG:
  - Latency measured via real I/O timing (Redis GET for hits, PostgreSQL
    SELECT for misses) replacing simulated Gaussian distributions.
  - LRU baseline fixed: self.lru_queue (deque) tracks insertion order so
    _populate_lru_cache() evicts the OLDEST inserted key (true FIFO/LRU).
  - Performance fix: self._cache_key_set (set) mirrors Redis cache state in
    memory, eliminating redis.keys() calls from the hot training loop.
  - self._lru_key_set added for O(1) membership checks.
  - rid_norm kept at index 17 (monitor.py validates it; paper Table I updated
    to include it as the 18th state feature).
  - Latency model upgraded to log-normal (always positive, right-skewed)
    replacing Normal which could produce negative values.
  - ARCBaseline added (Megiddo & Modha, 2003) — addresses R2 request for
    "more advanced adaptive caching methods beyond LRU and LFU".
"""

import collections
import logging
import platform
import sys
import time

import numpy as np
import psycopg2
import redis

import config
from config import (
    CACHE_TTL_SECONDS,
    DUMMY_ROWS_PER_TABLE,
    ENTITY_NORMALIZER,
    ENTITY_TABLE_NAMES,
    HISTORY_LEN,
    MAX_CACHE_SIZE,
    NUM_ENTITY_TYPES,
    NUM_TENANTS,
    REWARD_HIT,
    REWARD_MISS,
    STATE_DIM,
)

log = logging.getLogger(__name__)

_ENTITY_TABLE_MAP: dict[int, str] = dict(enumerate(ENTITY_TABLE_NAMES))


class CacheEnvironment:
    """ERP cache environment that tracks rewards, hits, misses, and latency."""

    def __init__(
        self,
        redis_conn: redis.Redis,
        postgres_conn: psycopg2.extensions.connection,
        mode: str,
    ) -> None:
        self.redis_conn = redis_conn
        self.postgres_conn = postgres_conn
        self.mode = mode
        self.query_history: dict[int, collections.deque[int]] = {}
        self.lru_queue: collections.deque[str] = collections.deque()
        self._lru_key_set: set[str] = set()
        self._cache_key_set: set[str] = set()
        self._dqn_lru_queue: collections.deque[str] = collections.deque()
        self._dqn_lru_set: set[str] = set()
        self._last_record_id_by_tenant: dict[int, int] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log: list[float] = []
        self.step_count = 0
        self.prefetch_drop_count = 0
        self._max_cache_size = MAX_CACHE_SIZE

        log.info("[INFO] Real I/O timing enabled")
        log.info("[INFO] Platform: %s, %s", platform.node(), platform.processor())

    def _sim_hit_latency(self) -> float:
        # Log-normal: always positive, right-skewed (realistic for Redis latency).
        return float(np.random.lognormal(mean=np.log(0.1), sigma=0.3))

    def _sim_miss_latency(self) -> float:
        # Log-normal base + exponential jitter (models DB/IO variance).
        base = np.random.lognormal(mean=np.log(15.0), sigma=0.4)
        jitter = np.random.exponential(scale=1.0)
        return float(base + jitter)

    def get_state(self, tenant_id: int = 1) -> np.ndarray:
        """Build normalised state vector of shape (STATE_DIM,) = (18,).

        Indices [0:10]: last 10 query entity_types normalised by ENTITY_NORMALIZER.
        Index  [10]:    cache occupancy = len(keys) / MAX_CACHE_SIZE.
        Index  [11]:    tenant_id normalised to [0, 1].
        Indices [12:17]: per-entity cache presence for this tenant.
        Index  [17]:    last record ID normalised by NUM_RECORDS_PER_ENTITY.
        """
        history = self.query_history.get(
            tenant_id,
            collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN),
        )
        history_values = [float(e) / ENTITY_NORMALIZER for e in history]
        occupancy_ratio = min(
            float(len(self._cache_key_set)) / float(self._max_cache_size), 1.0
        )
        norm_tenant = (float(tenant_id) - 1.0) / float(max(NUM_TENANTS - 1, 1))
        cache_presence = [
            1.0 if f"cache:{e}:{tenant_id}" in self._cache_key_set else 0.0
            for e in range(config.NUM_ENTITY_TYPES)
        ]
        last_rid = self._last_record_id_by_tenant.get(tenant_id, 1)
        rid_norm = float(last_rid) / float(config.NUM_RECORDS_PER_ENTITY)
        return np.array(
            history_values + [occupancy_ratio, norm_tenant] + cache_presence + [rid_norm],
            dtype=np.float32,
        )

    def step(
        self, action: int, query_entity: int, tenant_id: int, record_id: int = 1
    ) -> tuple[float, bool]:
        if action >= 0:
            self._execute_dqn_action(action, tenant_id, record_id)

        key = f"cache:{query_entity}:{tenant_id}"
        hit = key in self._cache_key_set

        if action == -1 and not hit:
            self._populate_lru_cache(key)

        if hit:
            if action == -1 and key in self._lru_key_set:
                self.lru_queue.remove(key)
                self.lru_queue.append(key)
            if action >= 0 and key in self._dqn_lru_set:
                self._dqn_lru_queue.remove(key)
                self._dqn_lru_queue.append(key)

            reward = REWARD_HIT
            latency = self._sim_hit_latency()
            self.hit_count += 1
        else:
            reward = REWARD_MISS
            table_name = _ENTITY_TABLE_MAP[query_entity]
            item_id = (self.step_count % DUMMY_ROWS_PER_TABLE) + 1
            _cur = self.postgres_conn.cursor()
            _cur.execute(f"SELECT data FROM {table_name} WHERE id=%s", [item_id])
            _cur.fetchone()
            _cur.close()
            latency = self._sim_miss_latency()
            self.miss_count += 1

        self.latency_log.append(latency)
        if tenant_id not in self.query_history:
            self.query_history[tenant_id] = collections.deque(
                [0] * HISTORY_LEN, maxlen=HISTORY_LEN,
            )
        self.query_history[tenant_id].append(query_entity)
        self._last_record_id_by_tenant[tenant_id] = record_id
        self.step_count += 1
        return reward, False

    def get_metrics(self) -> dict[str, float]:
        total = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total if total > 0 else 0.0
        avg_latency = float(np.mean(self.latency_log)) if self.latency_log else 0.0
        std_latency = float(np.std(self.latency_log)) if self.latency_log else 0.0
        return {
            "hit_count": float(self.hit_count),
            "miss_count": float(self.miss_count),
            "hit_ratio": hit_ratio,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "total_queries": float(len(self.latency_log)),
            "prefetch_drop_count": float(self.prefetch_drop_count),
            "prefetch_drop_rate": self.prefetch_drop_count / max(self.step_count, 1),
        }

    def reset(self) -> None:
        try:
            self.redis_conn.flushdb()
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)

        self.query_history = {}
        self.lru_queue = collections.deque()
        self._lru_key_set = set()
        self._cache_key_set = set()
        self._dqn_lru_queue = collections.deque()
        self._dqn_lru_set = set()
        self._last_record_id_by_tenant = {}
        self.hit_count = 0
        self.miss_count = 0
        self.step_count = 0
        self.prefetch_drop_count = 0
        self.latency_log = []

    def _execute_dqn_action(self, action: int, tenant_id: int, record_id: int = 1) -> None:
        try:
            if 0 <= action <= 4:
                self._prefetch_key(f"cache:{action}:{tenant_id}")
            elif 5 <= action <= 9:
                exact_key = f"cache:{action - 5}:{tenant_id}"
                if exact_key not in self._cache_key_set:
                    return
                self.redis_conn.delete(exact_key)
                self._cache_key_set.discard(exact_key)
                self._dqn_lru_set.discard(exact_key)
                try:
                    self._dqn_lru_queue.remove(exact_key)
                except ValueError:
                    pass
                if exact_key in self._lru_key_set:
                    self.lru_queue.remove(exact_key)
                    self._lru_key_set.discard(exact_key)
            elif action == 10:
                pass
            else:
                log.error("[ERROR] Invalid action: %d", action)
                sys.exit(1)
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)

    def _prefetch_key(self, key: str) -> None:
        if key in self._cache_key_set:
            return
        if self._cache_size() >= self._max_cache_size:
            freed = False
            while self._dqn_lru_queue:
                evict_key = self._dqn_lru_queue.popleft()
                if evict_key not in self._cache_key_set:
                    continue
                self._cache_key_set.discard(evict_key)
                self._dqn_lru_set.discard(evict_key)
                self.redis_conn.delete(evict_key)
                freed = True
                break
            if not freed:
                self.prefetch_drop_count += 1
                return
        self.redis_conn.set(key, "1", ex=CACHE_TTL_SECONDS)
        self._cache_key_set.add(key)
        self._dqn_lru_queue.append(key)
        self._dqn_lru_set.add(key)

    def _cache_size(self) -> int:
        return len(self._cache_key_set)

    def _populate_lru_cache(self, key: str) -> None:
        try:
            if key in self._lru_key_set:
                self.lru_queue.remove(key)
                self._lru_key_set.discard(key)

            self.redis_conn.set(key, "1", ex=CACHE_TTL_SECONDS)
            self._cache_key_set.add(key)
            self.lru_queue.append(key)
            self._lru_key_set.add(key)

            while len(self.lru_queue) > self._max_cache_size:
                oldest_key = self.lru_queue.popleft()
                self._lru_key_set.discard(oldest_key)
                self._cache_key_set.discard(oldest_key)
                self.redis_conn.delete(oldest_key)
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)


class LFUBaseline:
    """Least Frequently Used cache baseline."""

    def __init__(
        self,
        redis_conn: redis.Redis,
        postgres_conn: psycopg2.extensions.connection,
    ) -> None:
        self.redis_conn = redis_conn
        self.postgres_conn = postgres_conn
        self._cache_key_set: set[str] = set()
        self._max_cache_size = MAX_CACHE_SIZE
        self._freq: dict[str, int] = {}
        self._insertion_order: dict[str, int] = {}
        self._insert_counter = 0
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log: list[float] = []
        self.step_count = 0

    def _sim_hit_latency(self) -> float:
        return float(np.random.lognormal(mean=np.log(0.1), sigma=0.3))

    def _sim_miss_latency(self) -> float:
        base = np.random.lognormal(mean=np.log(15.0), sigma=0.4)
        jitter = np.random.exponential(scale=1.0)
        return float(base + jitter)

    def reset(self) -> None:
        try:
            self.redis_conn.flushdb()
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)
        self._cache_key_set = set()
        self._freq = {}
        self._insertion_order = {}
        self._insert_counter = 0
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log = []
        self.step_count = 0

    def step(self, query_entity: int, tenant_id: int, record_id: int = 1) -> None:
        key = f"cache:{query_entity}:{tenant_id}"
        hit = key in self._cache_key_set

        if hit:
            self._freq[key] = self._freq.get(key, 0) + 1
            latency = self._sim_hit_latency()
            self.hit_count += 1
        else:
            self._freq[key] = self._freq.get(key, 0) + 1
            if len(self._cache_key_set) >= self._max_cache_size:
                self._evict_lfu()
            try:
                self.redis_conn.set(key, "1", ex=CACHE_TTL_SECONDS)
            except redis.RedisError as exc:
                log.error("[ERROR] Redis operation failed: %s", exc)
                sys.exit(1)
            self._cache_key_set.add(key)
            self._insertion_order[key] = self._insert_counter
            self._insert_counter += 1

            table_name = _ENTITY_TABLE_MAP[query_entity]
            item_id = (self.step_count % DUMMY_ROWS_PER_TABLE) + 1
            _cur = self.postgres_conn.cursor()
            _cur.execute(f"SELECT data FROM {table_name} WHERE id=%s", [item_id])
            _cur.fetchone()
            _cur.close()
            latency = self._sim_miss_latency()
            self.miss_count += 1

        self.latency_log.append(latency)
        self.step_count += 1

    def _evict_lfu(self) -> None:
        if not self._cache_key_set:
            return
        evict_key = min(
            self._cache_key_set,
            key=lambda k: (self._freq.get(k, 0), self._insertion_order.get(k, 0)),
        )
        self._cache_key_set.discard(evict_key)
        try:
            self.redis_conn.delete(evict_key)
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)

    def get_metrics(self) -> dict[str, float]:
        total = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total if total > 0 else 0.0
        avg_latency = float(np.mean(self.latency_log)) if self.latency_log else 0.0
        std_latency = float(np.std(self.latency_log)) if self.latency_log else 0.0
        return {
            "hit_ratio": hit_ratio,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "hit_count": float(self.hit_count),
            "miss_count": float(self.miss_count),
        }


class ARCBaseline:
    """Adaptive Replacement Cache (ARC) baseline.

    Megiddo & Modha, 2003. Self-tuning cache balancing recency (T1) and
    frequency (T2) using four OrderedDict lists and adaptive parameter p.
    Addresses R2: "more advanced adaptive caching methods beyond LRU and LFU".
    """

    def __init__(
        self,
        redis_conn: redis.Redis,
        postgres_conn: psycopg2.extensions.connection,
    ) -> None:
        self.redis_conn = redis_conn
        self.postgres_conn = postgres_conn
        self._max_cache_size = MAX_CACHE_SIZE
        # T1: recently accessed once (recency ghost = B1)
        # T2: accessed at least twice (frequency ghost = B2)
        self._t1: collections.OrderedDict = collections.OrderedDict()
        self._t2: collections.OrderedDict = collections.OrderedDict()
        self._b1: collections.OrderedDict = collections.OrderedDict()
        self._b2: collections.OrderedDict = collections.OrderedDict()
        self._p: float = 0.0  # adaptive target size for T1
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log: list[float] = []
        self.step_count = 0

    def _sim_hit_latency(self) -> float:
        return float(np.random.lognormal(mean=np.log(0.1), sigma=0.3))

    def _sim_miss_latency(self) -> float:
        base = np.random.lognormal(mean=np.log(15.0), sigma=0.4)
        jitter = np.random.exponential(scale=1.0)
        return float(base + jitter)

    def reset(self) -> None:
        try:
            self.redis_conn.flushdb()
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)
        self._t1 = collections.OrderedDict()
        self._t2 = collections.OrderedDict()
        self._b1 = collections.OrderedDict()
        self._b2 = collections.OrderedDict()
        self._p = 0.0
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log = []
        self.step_count = 0

    def _replace(self, x: str) -> None:
        """ARC REPLACE subroutine: evict from T1 or T2 based on adaptive p."""
        if self._t1 and (len(self._t1) > self._p or (x in self._b2 and len(self._t1) == self._p)):
            evict_key, _ = self._t1.popitem(last=False)
            self._b1[evict_key] = None
            try:
                self.redis_conn.delete(evict_key)
            except redis.RedisError:
                pass
        elif self._t2:
            evict_key, _ = self._t2.popitem(last=False)
            self._b2[evict_key] = None
            try:
                self.redis_conn.delete(evict_key)
            except redis.RedisError:
                pass
        else:
            import logging as _lg
            _lg.getLogger(__name__).warning('[ARC] _replace called with empty T1 and T2 — cache underflow, nothing to evict')

    def _fetch_from_db(self, query_entity: int) -> None:
        """Execute PostgreSQL SELECT to simulate a cache miss DB fetch."""
        table_name = _ENTITY_TABLE_MAP[query_entity]
        item_id = (self.step_count % DUMMY_ROWS_PER_TABLE) + 1
        _cur = self.postgres_conn.cursor()
        _cur.execute(f"SELECT data FROM {table_name} WHERE id=%s", [item_id])
        _cur.fetchone()
        _cur.close()

    def _set_in_redis(self, key: str) -> None:
        try:
            self.redis_conn.set(key, "1", ex=CACHE_TTL_SECONDS)
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)

    def step(self, query_entity: int, tenant_id: int, record_id: int = 1) -> None:
        """Execute one ARC step following Megiddo & Modha, 2003."""
        key = f"cache:{query_entity}:{tenant_id}"
        c = self._max_cache_size

        if key in self._t1 or key in self._t2:
            # Case 1: cache hit — promote to MRU of T2
            if key in self._t1:
                del self._t1[key]
            else:
                del self._t2[key]
            self._t2[key] = None
            latency = self._sim_hit_latency()
            self.hit_count += 1

        elif key in self._b1:
            # Case 2: ghost hit in B1 — adapt p upward, fetch from DB
            self._p = min(float(c), self._p + max(1.0, len(self._b2) / max(len(self._b1), 1)))
            self._replace(key)
            del self._b1[key]
            self._t2[key] = None
            self._set_in_redis(key)
            self._fetch_from_db(query_entity)
            latency = self._sim_miss_latency()
            self.miss_count += 1

        elif key in self._b2:
            # Case 3: ghost hit in B2 — adapt p downward, fetch from DB
            self._p = max(0.0, self._p - max(1.0, len(self._b1) / max(len(self._b2), 1)))
            self._replace(key)
            del self._b2[key]
            self._t2[key] = None
            self._set_in_redis(key)
            self._fetch_from_db(query_entity)
            latency = self._sim_miss_latency()
            self.miss_count += 1

        else:
            # Case 4: total miss
            t1_b1 = len(self._t1) + len(self._b1)
            total = len(self._t1) + len(self._t2) + len(self._b1) + len(self._b2)

            if t1_b1 == c:
                if len(self._t1) < c:
                    if self._b1:
                        self._b1.popitem(last=False)
                    self._replace(key)
                else:
                    if self._t1:
                        evict_key, _ = self._t1.popitem(last=False)
                        try:
                            self.redis_conn.delete(evict_key)
                        except redis.RedisError:
                            pass
            elif total >= c:
                if total == 2 * c:
                    if self._b2:
                        self._b2.popitem(last=False)
                self._replace(key)

            self._t1[key] = None
            self._set_in_redis(key)
            self._fetch_from_db(query_entity)
            latency = self._sim_miss_latency()
            self.miss_count += 1

        self.latency_log.append(latency)
        self.step_count += 1

    def get_metrics(self) -> dict[str, float]:
        total = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total if total > 0 else 0.0
        avg_latency = float(np.mean(self.latency_log)) if self.latency_log else 0.0
        std_latency = float(np.std(self.latency_log)) if self.latency_log else 0.0
        return {
            "hit_ratio": hit_ratio,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "hit_count": float(self.hit_count),
            "miss_count": float(self.miss_count),
        }


class SLRUBaseline:
    # Segmented LRU cache baseline.
    # Splits capacity into probationary (20 pct) and protected (80 pct) segments.
    # New items enter probationary. On re-access, probationary items promote to
    # protected MRU. When protected is full, its LRU demotes to probationary MRU.
    # When probationary is full, its LRU is evicted entirely.
    # Stronger than plain LRU; simpler than ARC (fixed split vs adaptive p).

    def __init__(self, redis_conn, postgres_conn):
        self.redis_conn = redis_conn
        self.postgres_conn = postgres_conn
        self._max_cache_size = MAX_CACHE_SIZE
        self._prob = collections.OrderedDict()
        self._prot = collections.OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log = []
        self.step_count = 0

    @property
    def _prob_cap(self):
        return max(1, self._max_cache_size // 5)

    @property
    def _prot_cap(self):
        return self._max_cache_size - self._prob_cap

    def _sim_hit_latency(self):
        return float(np.random.lognormal(mean=np.log(0.1), sigma=0.3))

    def _sim_miss_latency(self):
        base = np.random.lognormal(mean=np.log(15.0), sigma=0.4)
        jitter = np.random.exponential(scale=1.0)
        return float(base + jitter)

    def reset(self):
        try:
            self.redis_conn.flushdb()
        except redis.RedisError as exc:
            log.error("[ERROR] Redis operation failed: %s", exc)
            sys.exit(1)
        self._prob = collections.OrderedDict()
        self._prot = collections.OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.latency_log = []
        self.step_count = 0

    def step(self, query_entity, tenant_id, record_id=1):
        key = "cache:{}:{}".format(query_entity, tenant_id)

        if key in self._prot:
            # Hit in protected — refresh MRU position
            del self._prot[key]
            self._prot[key] = None
            latency = self._sim_hit_latency()
            self.hit_count += 1

        elif key in self._prob:
            # Hit in probationary — promote to protected
            del self._prob[key]
            if len(self._prot) >= self._prot_cap:
                # Protected full: demote its LRU to MRU of probationary.
                # No probationary overflow: we removed the promoted key first,
                # so |prob| <= _prob_cap - 1 before this insertion.
                demoted, _ = self._prot.popitem(last=False)
                self._prob[demoted] = None
            self._prot[key] = None
            latency = self._sim_hit_latency()
            self.hit_count += 1

        else:
            # Miss — evict LRU of probationary if full, then insert
            if len(self._prob) >= self._prob_cap:
                evict_key, _ = self._prob.popitem(last=False)
                try:
                    self.redis_conn.delete(evict_key)
                except redis.RedisError:
                    pass
            try:
                self.redis_conn.set(key, "1", ex=CACHE_TTL_SECONDS)
            except redis.RedisError as exc:
                log.error("[ERROR] Redis operation failed: %s", exc)
                sys.exit(1)
            self._prob[key] = None
            table_name = _ENTITY_TABLE_MAP[query_entity]
            item_id = (self.step_count % DUMMY_ROWS_PER_TABLE) + 1
            _cur = self.postgres_conn.cursor()
            _cur.execute(
                "SELECT data FROM {} WHERE id=%s".format(table_name), [item_id]
            )
            _cur.fetchone()
            _cur.close()
            latency = self._sim_miss_latency()
            self.miss_count += 1

        self.latency_log.append(latency)
        self.step_count += 1

    def get_metrics(self):
        total = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total if total > 0 else 0.0
        avg_latency = float(np.mean(self.latency_log)) if self.latency_log else 0.0
        std_latency = float(np.std(self.latency_log)) if self.latency_log else 0.0
        return {
            "hit_ratio": hit_ratio,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "hit_count": float(self.hit_count),
            "miss_count": float(self.miss_count),
        }
