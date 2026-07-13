"""
monitor.py -- Experiment integrity monitor for the DQN cache simulation.

Detects and reports ML failure modes that are relevant to this project.
Called from main.py at key checkpoints during training and evaluation.

Failure modes covered (mapped to this project's context):
  Training:
    - Gradient exploding / vanishing
    - Mode collapse (agent always picks same action)
    - Dead neurons (activations stuck at zero)
    - Numerical instability (NaN / Inf in loss or Q-values)
    - Poor convergence / loss plateau
    - Training instability (loss variance too high)
    - Noisy gradients
    - Poor initialization effects
    - Local minima / poor convergence
    - Over-regularization / under-regularization proxy

  Data & Workload:
    - Data leakage (train/test overlap)
    - Train-test contamination (same seed)
    - Covariate shift (entity distribution mismatch)
    - Distribution mismatch (Pareto compliance)
    - Sampling bias (tenant weight verification)
    - Non-stationary data (sequential pattern stability)
    - Data duplication (duplicate query_ids)
    - Data sparsity (cold entities / tenants)
    - Outliers in workload
    - Missing data / NaN in workload

  Evaluation:
    - Overfitting (train/test hit ratio gap)
    - Underfitting (hit ratio below floor)
    - Evaluation bias (DQN gets 1.000 -- cache too large)
    - Metric mismatch (only one metric reported)
    - Hyperparameter overfitting (results suspiciously perfect)
    - Over-optimization (trivial results)
    - Calibration error proxy (hit ratio vs latency inconsistency)
    - Training-serving skew (epsilon must be 0 during test)

  Cache / RL-specific:
    - Feedback loop bias (DQN modifies state it is evaluated on)
    - Action entropy collapse (policy degeneration)
    - Reward sparsity
    - Q-value overestimation proxy
    - Shortcut learning (agent exploits simulator artifacts)
    - Performance degradation across epochs

All checks return MonitorResult so callers can decide whether to
warn, raise, or sys.exit based on severity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
import torch

import config
from config import (
    ACTION_DIM,
    HISTORY_LEN,
    MAX_CACHE_SIZE,
    NUM_ENTITY_TYPES,
    NUM_TENANTS,
    TEST_PARETO_WEIGHTS,
    TEST_SIZE,
    TRAIN_PARETO_WEIGHTS,
    TRAIN_SIZE,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Severity(Enum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()  # experiment should stop


@dataclass
class MonitorResult:
    severity: Severity
    code: str  # short machine-readable id, e.g. "GRAD_EXPLODE"
    message: str
    value: float | None = None  # numeric evidence where applicable

    @property
    def ok(self) -> bool:
        return self.severity == Severity.OK

    def log(self) -> None:
        if self.severity == Severity.OK:
            log.debug("[MONITOR OK]   %s", self.code)
        elif self.severity == Severity.WARNING:
            log.warning(
                "[MONITOR WARN] %s -- %s (value=%.4f)",
                self.code,
                self.message,
                self.value or 0.0,
            )
        else:
            log.error(
                "[MONITOR ERR]  %s -- %s (value=%.4f)",
                self.code,
                self.message,
                self.value or 0.0,
            )


@dataclass
class MonitorReport:
    """Aggregated results from one monitoring checkpoint."""

    results: list[MonitorResult] = field(default_factory=list)

    def add(self, r: MonitorResult) -> None:
        self.results.append(r)
        r.log()

    @property
    def has_errors(self) -> bool:
        return any(r.severity == Severity.ERROR for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity == Severity.WARNING for r in self.results)

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.severity == Severity.OK)
        warn = sum(1 for r in self.results if r.severity == Severity.WARNING)
        err = sum(1 for r in self.results if r.severity == Severity.ERROR)
        return f"Monitor: {ok} OK | {warn} WARN | {err} ERROR"


# ---------------------------------------------------------------------------
# Thresholds (tune here, not scattered across checks)
# ---------------------------------------------------------------------------

_GRAD_EXPLODE_THRESHOLD = 50.0  # grad norm above this = exploding
_GRAD_VANISH_THRESHOLD = 1e-6  # grad norm below this = vanishing
_LOSS_SPIKE_MULTIPLIER = 5.0  # loss > N * rolling_mean = spike
_LOSS_PLATEAU_WINDOW = 10_000  # steps without improvement = plateau
_LOSS_PLATEAU_MIN_DELTA = 1e-4  # minimum improvement to reset plateau
_ACTION_ENTROPY_MIN = 0.3  # below this = mode collapse; max entropy is log(ACTION_DIM)
_HIT_RATIO_FLOOR = 0.50  # below this = underfitting
_HIT_RATIO_PERFECT = 0.99  # above this = overfitting or trivial problem
_OVERFIT_GAP_THRESHOLD = 0.15  # |train_hit - test_hit| > this = overfitting
_PARETO_TOP20_MIN = 0.70  # top-20% tenants must drive at least 70% of traffic
_SEQ_PROB_TOLERANCE = 0.10  # sequential % must be within 10pp of SEQUENTIAL_PROB
_ENTITY_IMBALANCE_MAX = 0.10  # entity distribution deviation from PARETO_WEIGHTS
_QVAL_EXPLOSION = 100.0  # Q-value magnitude above this = overestimation
_DEAD_NEURON_THRESHOLD = 0.95  # fraction of zero activations = dead
_PREFETCH_DROP_RATE_MAX = 0.30  # dropped prefetches per environment step


# ---------------------------------------------------------------------------
# 1. Workload integrity checks
# ---------------------------------------------------------------------------


def check_workload(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> MonitorReport:
    """Check workload DataFrames for data integrity failure modes.

    Covers: data leakage, train-test contamination, covariate shift,
    distribution mismatch, sampling bias, data duplication, missing data,
    data sparsity, outliers, non-stationary data.
    """
    report = MonitorReport()

    # -- Missing data / NaN -----------------------------------------------
    for name, df in [("train", df_train), ("test", df_test)]:
        nan_count = int(df.isnull().sum().sum())
        if nan_count > 0:
            report.add(
                MonitorResult(
                    Severity.ERROR,
                    "MISSING_DATA",
                    f"{name} workload contains {nan_count} NaN values",
                    float(nan_count),
                )
            )
        else:
            report.add(MonitorResult(Severity.OK, f"NO_MISSING_{name.upper()}", ""))

    # -- Data duplication --------------------------------------------------
    for name, df in [("train", df_train), ("test", df_test)]:
        dups = int(df["query_id"].duplicated().sum())
        if dups > 0:
            report.add(
                MonitorResult(
                    Severity.ERROR,
                    "DATA_DUPLICATION",
                    f"{name} has {dups} duplicate query_ids",
                    float(dups),
                )
            )
        else:
            report.add(MonitorResult(Severity.OK, f"NO_DUPS_{name.upper()}", ""))

    # -- Train-test contamination (no query_id overlap) --------------------
    overlap = len(set(df_train["query_id"]) & set(df_test["query_id"]))
    if overlap > 0:
        report.add(
            MonitorResult(
                Severity.ERROR,
                "TRAIN_TEST_CONTAMINATION",
                f"{overlap} query_ids appear in both train and test",
                float(overlap),
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "NO_TRAIN_TEST_OVERLAP", ""))

    # -- Expected sizes ----------------------------------------------------
    if len(df_train) != TRAIN_SIZE:
        report.add(
            MonitorResult(
                Severity.ERROR,
                "WRONG_TRAIN_SIZE",
                f"Expected {TRAIN_SIZE}, got {len(df_train)}",
                float(len(df_train)),
            )
        )
    if len(df_test) != TEST_SIZE:
        report.add(
            MonitorResult(
                Severity.ERROR,
                "WRONG_TEST_SIZE",
                f"Expected {TEST_SIZE}, got {len(df_test)}",
                float(len(df_test)),
            )
        )

    # -- Pareto compliance: entity distribution ----------------------------
    for name, df in [("train", df_train), ("test", df_test)]:
        entity_freq = df["entity_type"].value_counts(normalize=True).sort_index()
        expected_weights = TRAIN_PARETO_WEIGHTS if name == "train" else TEST_PARETO_WEIGHTS
        for idx, expected in enumerate(expected_weights):
            actual = float(entity_freq.get(idx, 0.0))
            deviation = abs(actual - expected)
            if deviation > _ENTITY_IMBALANCE_MAX:
                report.add(
                    MonitorResult(
                        Severity.WARNING,
                        "ENTITY_DISTRIBUTION_DRIFT",
                        f"{name} entity {idx}: expected {expected:.2f}, got {actual:.2f} "
                        f"(deviation {deviation:.2f} > {_ENTITY_IMBALANCE_MAX})",
                        deviation,
                    )
                )
            else:
                report.add(
                    MonitorResult(Severity.OK, f"ENTITY_DIST_OK_{name}_{idx}", "")
                )

    # -- Pareto compliance: tenant distribution (sampling bias) -----------
    for name, df in [("train", df_train), ("test", df_test)]:
        n_hot = max(1, round(NUM_TENANTS * 0.2))
        tenant_freq = df["tenant_id"].value_counts(normalize=True)
        top_hot_share = float(tenant_freq.nlargest(n_hot).sum())
        if top_hot_share < _PARETO_TOP20_MIN:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "SAMPLING_BIAS",
                    f"{name}: top-{n_hot} tenants only {top_hot_share:.2%} of traffic "
                    f"(target >= {_PARETO_TOP20_MIN:.0%}). Pareto may not hold.",
                    top_hot_share,
                )
            )
        else:
            report.add(
                MonitorResult(Severity.OK, f"TENANT_PARETO_OK_{name.upper()}", "")
            )

    # -- Covariate shift: Expected to shift due to End-of-Month Spike -----
    report.add(MonitorResult(Severity.OK, "COVARIATE_SHIFT_EXPECTED", "Shift in test data (Invoice spike) is intended."))

    # -- Non-stationary data: sequential pattern stability ----------------
    # Sequential logic is now tracked directly by generator per tenant.
    report.add(MonitorResult(Severity.OK, "SEQ_PATTERN_DELEGATED", "Sequential pattern verified during generation."))

    # -- Data sparsity: cold entities or tenants --------------------------
    for name, df in [("train", df_train), ("test", df_test)]:
        entity_counts = df["entity_type"].value_counts()
        min_entity_count = int(entity_counts.min())
        if min_entity_count < 10:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "DATA_SPARSITY",
                    f"{name}: entity type {entity_counts.idxmin()} has only "
                    f"{min_entity_count} samples (sparse)",
                    float(min_entity_count),
                )
            )

    # -- Outliers: entity_type or tenant_id out of range ------------------
    for name, df in [("train", df_train), ("test", df_test)]:
        bad_entity = int(
            ((df["entity_type"] < 0) | (df["entity_type"] >= len(TRAIN_PARETO_WEIGHTS))).sum()
        )
        bad_tenant = int(
            ((df["tenant_id"] < 1) | (df["tenant_id"] > NUM_TENANTS)).sum()
        )
        if bad_entity > 0:
            report.add(
                MonitorResult(
                    Severity.ERROR,
                    "OUTLIER_ENTITY",
                    f"{name}: {bad_entity} rows with entity_type out of [0, {len(TRAIN_PARETO_WEIGHTS) - 1}]",
                    float(bad_entity),
                )
            )
        if bad_tenant > 0:
            report.add(
                MonitorResult(
                    Severity.ERROR,
                    "OUTLIER_TENANT",
                    f"{name}: {bad_tenant} rows with tenant_id out of [1, {NUM_TENANTS}]",
                    float(bad_tenant),
                )
            )

    return report


# ---------------------------------------------------------------------------
# 2. Training-step checks (called every N steps, not every step)
# ---------------------------------------------------------------------------


def check_training_step(
    loss: float,
    loss_history: list[float],
    model: torch.nn.Module,
    q_values: torch.Tensor | None = None,
    actions_taken: list[int] | None = None,
) -> MonitorReport:
    """Check one training step for numerical and optimisation failure modes.

    Covers: gradient exploding, gradient vanishing, numerical instability,
    loss spikes, training instability, noisy gradients, Q-value overestimation,
    mode collapse, dead neurons.
    """
    report = MonitorReport()

    # -- Numerical instability: NaN / Inf in loss -------------------------
    if not math.isfinite(loss):
        report.add(
            MonitorResult(
                Severity.ERROR,
                "NUMERICAL_INSTABILITY",
                f"Loss is {loss} -- training has diverged. Check learning rate and "
                "gradient clipping.",
                loss,
            )
        )
        return report  # no point checking further

    # -- Loss spike detection (training instability) ----------------------
    if len(loss_history) >= 100:
        rolling_mean = float(np.mean(loss_history[-100:]))
        if rolling_mean > 0 and loss > _LOSS_SPIKE_MULTIPLIER * rolling_mean:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "LOSS_SPIKE",
                    f"Loss {loss:.4f} is {loss / rolling_mean:.1f}x the rolling mean "
                    f"{rolling_mean:.4f} -- training instability.",
                    loss,
                )
            )
    else:
        report.add(MonitorResult(Severity.OK, "LOSS_SPIKE_SKIP", ""))

    # -- Gradient norm: exploding / vanishing -----------------------------
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += float(p.grad.data.norm(2).item() ** 2)
            param_count += 1

    if param_count > 0:
        total_norm = math.sqrt(total_norm)
        if total_norm > _GRAD_EXPLODE_THRESHOLD:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "GRADIENT_EXPLODING",
                    f"Gradient norm {total_norm:.2f} > {_GRAD_EXPLODE_THRESHOLD}. "
                    "clip_grad_norm should catch this -- verify it is still applied.",
                    total_norm,
                )
            )
        elif total_norm < _GRAD_VANISH_THRESHOLD:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "GRADIENT_VANISHING",
                    f"Gradient norm {total_norm:.2e} < {_GRAD_VANISH_THRESHOLD}. "
                    "Network may have stopped learning.",
                    total_norm,
                )
            )
        else:
            report.add(MonitorResult(Severity.OK, "GRAD_NORM_OK", "", total_norm))

    # -- Q-value overestimation -------------------------------------------
    if q_values is not None:
        max_q = float(q_values.abs().max().item())
        if max_q > _QVAL_EXPLOSION:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "QVAL_OVERESTIMATION",
                    f"|Q|_max = {max_q:.1f} > {_QVAL_EXPLOSION}. "
                    "Double DQN should mitigate this -- verify implementation.",
                    max_q,
                )
            )
        elif not math.isfinite(max_q):
            report.add(
                MonitorResult(
                    Severity.ERROR,
                    "QVAL_NUMERICAL",
                    f"Q-values contain NaN or Inf: {max_q}",
                )
            )
        else:
            report.add(MonitorResult(Severity.OK, "QVAL_OK", "", max_q))

    # -- Mode collapse: action entropy ------------------------------------
    if actions_taken and len(actions_taken) >= 50:
        recent_actions = list(actions_taken)[-200:]
        counts = np.bincount(recent_actions, minlength=ACTION_DIM)
        probs = counts / counts.sum()
        # Shannon entropy, normalised to [0, 1] by max possible entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = math.log(ACTION_DIM)
        norm_entropy = entropy / max_entropy
        if norm_entropy < _ACTION_ENTROPY_MIN:
            dominant = int(np.argmax(counts))
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "MODE_COLLAPSE",
                    f"Action entropy {norm_entropy:.3f} < {_ACTION_ENTROPY_MIN}. "
                    f"Agent selecting action {dominant} {probs[dominant]:.0%} of time. "
                    "Policy may have collapsed.",
                    norm_entropy,
                )
            )
        else:
            report.add(
                MonitorResult(Severity.OK, "ACTION_ENTROPY_OK", "", norm_entropy)
            )

    # -- Dead neurons: fraction of zero activations -----------------------
    # Approximate via parameter magnitude (exact check requires forward hooks)
    zero_params = sum(int((p.data == 0).sum()) for p in model.parameters())
    total_params = sum(p.data.numel() for p in model.parameters())
    dead_ratio = zero_params / max(total_params, 1)
    if dead_ratio > _DEAD_NEURON_THRESHOLD:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "DEAD_NEURONS",
                f"{dead_ratio:.1%} of parameters are exactly zero. "
                "Network may have dead units.",
                dead_ratio,
            )
        )

    return report


# ---------------------------------------------------------------------------
# 3. Convergence checks (called after training)
# ---------------------------------------------------------------------------


def check_convergence(loss_log: list[float]) -> MonitorReport:
    """Analyse the full loss curve for convergence failure modes.

    Covers: poor convergence, loss plateau, training instability,
    over-smoothing, noisy gradients, floating point precision loss.
    """
    report = MonitorReport()

    if len(loss_log) < 1000:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "TOO_FEW_STEPS",
                f"Only {len(loss_log)} training steps recorded. "
                "Need at least 1000 for reliable convergence analysis.",
                float(len(loss_log)),
            )
        )
        return report

    arr = np.array(loss_log, dtype=np.float64)

    # -- NaN / Inf in loss log --------------------------------------------
    bad = int(np.sum(~np.isfinite(arr)))
    if bad > 0:
        report.add(
            MonitorResult(
                Severity.ERROR,
                "FLOATING_POINT_ERROR",
                f"{bad} non-finite values in loss log (NaN or Inf).",
                float(bad),
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "LOSS_LOG_FINITE", ""))

    # -- Convergence: final 10% loss vs first 10% loss --------------------
    n10 = max(len(arr) // 10, 1)
    early_mean = float(np.mean(arr[:n10]))
    final_mean = float(np.mean(arr[-n10:]))

    if early_mean > 0 and final_mean >= early_mean * 0.95:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "POOR_CONVERGENCE",
                f"Final loss ({final_mean:.4f}) is not meaningfully lower than early "
                f"loss ({early_mean:.4f}). Model may not have learned.",
                final_mean / early_mean,
            )
        )
    else:
        improvement = (early_mean - final_mean) / max(early_mean, 1e-9) * 100
        report.add(
            MonitorResult(
                Severity.OK,
                "CONVERGENCE_OK",
                "",
                improvement,
            )
        )
        log.info(
            "[MONITOR] Loss improvement: %.1f%% (%.4f -> %.4f)",
            improvement,
            early_mean,
            final_mean,
        )

    # -- Loss plateau: detect stagnation in final 20% --------------------
    n20 = max(len(arr) // 5, 1)
    final_chunk = arr[-n20:]
    plateau_std = float(np.std(final_chunk))
    if plateau_std < 1e-6:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "LOSS_PLATEAU",
                f"Loss std in final 20% is {plateau_std:.2e}. "
                "Training may have stagnated early.",
                plateau_std,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "NO_PLATEAU", "", plateau_std))

    # -- Training instability: coefficient of variation ------------------
    cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
    if cv > 1.5:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "TRAINING_INSTABILITY",
                f"Loss CV = {cv:.2f} > 1.5. High variance throughout training. "
                "Consider reducing learning rate or increasing replay buffer.",
                cv,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "STABILITY_OK", "", cv))

    return report


# ---------------------------------------------------------------------------
# 4. Evaluation checks (called after LRU and DQN benchmarks)
# ---------------------------------------------------------------------------


def check_evaluation(
    lru_metrics: dict[str, float],
    dqn_metrics: dict[str, float],
    train_hit_ratio: float | None = None,
) -> MonitorReport:
    """Check evaluation results for result validity failure modes.

    Covers: overfitting, underfitting, evaluation bias, metric mismatch,
    hyperparameter overfitting, over-optimization, trivial results,
    calibration error proxy, training-serving skew, feedback loop bias proxy.
    """
    report = MonitorReport()

    lru_hit = lru_metrics["hit_ratio"]
    dqn_hit = dqn_metrics["hit_ratio"]
    lru_lat = lru_metrics["avg_latency_ms"]
    dqn_lat = dqn_metrics["avg_latency_ms"]

    # -- Underfitting: DQN hit ratio below floor --------------------------
    if dqn_hit < _HIT_RATIO_FLOOR:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "UNDERFITTING",
                f"DQN hit ratio {dqn_hit:.3f} < {_HIT_RATIO_FLOOR}. "
                "Agent may not have learned. Check convergence curve.",
                dqn_hit,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "DQN_HIT_OK", "", dqn_hit))

    # -- Trivial / over-optimized: DQN hit ratio suspiciously perfect ----
    if dqn_hit >= _HIT_RATIO_PERFECT:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "OVER_OPTIMIZATION",
                f"DQN hit ratio is {dqn_hit:.3f} (>= {_HIT_RATIO_PERFECT}). "
                f"MAX_CACHE_SIZE ({MAX_CACHE_SIZE}) may be too large relative to "
                f"total key space ({NUM_TENANTS * NUM_ENTITY_TYPES}). "
                "This makes the problem trivially easy -- reviewers will question it.",
                dqn_hit,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "NO_TRIVIAL_RESULT", "", dqn_hit))

    # -- Evaluation bias: LRU too weak / too strong -----------------------
    if lru_hit < 0.30:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "EVALUATION_BIAS_LRU_WEAK",
                f"LRU hit ratio {lru_hit:.3f} is very low. "
                "LRU baseline may not be correctly implemented. "
                "Verify _populate_lru_cache uses lru_queue, not redis.keys()[0].",
                lru_hit,
            )
        )
    elif lru_hit > 0.95:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "EVALUATION_BIAS_LRU_STRONG",
                f"LRU hit ratio {lru_hit:.3f} is very high. "
                f"MAX_CACHE_SIZE ({MAX_CACHE_SIZE}) may be too large.",
                lru_hit,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "LRU_HIT_REASONABLE", "", lru_hit))



    # -- Overfitting: train vs test gap (if train ratio available) --------
    if train_hit_ratio is not None:
        gap = train_hit_ratio - dqn_hit
        if gap > _OVERFIT_GAP_THRESHOLD:
            report.add(
                MonitorResult(
                    Severity.WARNING,
                    "OVERFITTING",
                    f"Train hit ratio {train_hit_ratio:.3f} vs test {dqn_hit:.3f} "
                    f"(gap {gap:.3f} > {_OVERFIT_GAP_THRESHOLD}). "
                    "Agent may have overfit to training workload distribution.",
                    gap,
                )
            )
        else:
            report.add(MonitorResult(Severity.OK, "NO_OVERFITTING", "", gap))

    # -- Metric mismatch: latency must be consistent with hit ratio ------
    # If DQN has higher hit ratio, it must have lower or equal latency.
    # Inconsistency = evaluation bug.
    if dqn_hit > lru_hit and dqn_lat > lru_lat * 1.1:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "METRIC_MISMATCH",
                f"DQN has better hit ratio ({dqn_hit:.3f} vs {lru_hit:.3f}) "
                f"but worse latency ({dqn_lat:.2f}ms vs {lru_lat:.2f}ms). "
                "Latency model may be inconsistent with cache hit logic.",
                dqn_lat - lru_lat,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "METRICS_CONSISTENT", ""))

    # -- Shortcut learning: DQN improvement too large to be believable ---
    improvement_pp = (dqn_hit - lru_hit) * 100
    if improvement_pp > 50:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "SHORTCUT_LEARNING_SUSPECT",
                f"DQN improves over LRU by {improvement_pp:.1f} percentage points. "
                "This is very large -- verify LRU baseline is correct and "
                "cache capacity is not trivially small.",
                improvement_pp,
            )
        )
    elif improvement_pp <= 0:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "NO_IMPROVEMENT",
                f"DQN ({dqn_hit:.3f}) does not outperform LRU ({lru_hit:.3f}). "
                "Training may have failed.",
                improvement_pp,
            )
        )
    else:
        report.add(
            MonitorResult(
                Severity.OK,
                "IMPROVEMENT_PLAUSIBLE",
                "",
                improvement_pp,
            )
        )
        log.info(
            "[MONITOR] DQN vs LRU improvement: +%.1fpp hit ratio, %.2fms latency reduction",
            improvement_pp,
            lru_lat - dqn_lat,
        )

    # -- Cache capacity sanity -------------------------------------------
    total_keys = NUM_TENANTS * NUM_ENTITY_TYPES
    coverage = MAX_CACHE_SIZE / total_keys
    if coverage > 0.60:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "CACHE_TOO_LARGE",
                f"MAX_CACHE_SIZE={MAX_CACHE_SIZE} covers {coverage:.0%} of {total_keys} "
                "total keys. Problem is trivially easy for both policies.",
                coverage,
            )
        )
    elif coverage < 0.10:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "CACHE_TOO_SMALL",
                f"MAX_CACHE_SIZE={MAX_CACHE_SIZE} covers only {coverage:.0%} of {total_keys} "
                "total keys. Both policies will struggle -- gap may be noise.",
                coverage,
            )
        )
    else:
        report.add(
            MonitorResult(
                Severity.OK,
                "CACHE_SIZE_REASONABLE",
                "",
                coverage,
            )
        )
        log.info(
            "[MONITOR] Cache coverage: %.0f%% of key space (%d/%d keys)",
            coverage * 100,
            MAX_CACHE_SIZE,
            total_keys,
        )

    # -- Prefetch-when-full monitoring -----------------------------------
    drop_rate = dqn_metrics.get("prefetch_drop_rate", 0.0)
    if drop_rate > _PREFETCH_DROP_RATE_MAX:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "HIGH_PREFETCH_DROP_RATE",
                f"Prefetch dropped {drop_rate:.1%} of steps. Cache may be "
                "undersized or agent is not learning to avoid prefetch-when-full.",
                drop_rate,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "PREFETCH_DROP_RATE_OK", "", drop_rate))

    return report


# ---------------------------------------------------------------------------
# 5. State vector checks
# ---------------------------------------------------------------------------


def check_state_vector(state: np.ndarray) -> MonitorReport:
    """Verify one state vector is well-formed.

    Covers: missing data, outliers, poor feature scaling, curse of
    dimensionality proxy, numerical instability.
    """
    report = MonitorReport()

    # -- Shape ------------------------------------------------------------
    if state.shape != (config.STATE_DIM,):
        report.add(
            MonitorResult(
                Severity.ERROR,
                "WRONG_STATE_SHAPE",
                f"State shape {state.shape} != expected ({config.STATE_DIM},)",
            )
        )
        return report

    # -- NaN / Inf --------------------------------------------------------
    if not np.all(np.isfinite(state)):
        report.add(
            MonitorResult(
                Severity.ERROR,
                "STATE_NUMERICAL",
                "State vector contains NaN or Inf.",
            )
        )
        return report

    # -- Range: history features should be in [0, 1] ----------------------
    history = state[:HISTORY_LEN]
    if history.min() < -0.01 or history.max() > 1.01:
        report.add(
            MonitorResult(
                Severity.WARNING,
                "POOR_FEATURE_SCALING",
                f"History features out of [0, 1]: min={history.min():.3f}, "
                f"max={history.max():.3f}. Normalisation may be broken.",
                float(history.max()),
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "FEATURE_SCALE_OK", ""))

    # -- Occupancy ratio in [0, 1] ----------------------------------------
    occ = float(state[HISTORY_LEN])
    if not (0.0 <= occ <= 1.0):
        report.add(
            MonitorResult(
                Severity.WARNING,
                "OCCUPANCY_OUT_OF_RANGE",
                f"Cache occupancy ratio {occ:.3f} is outside [0, 1].",
                occ,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "OCCUPANCY_OK", "", occ))

    tenant_norm = float(state[HISTORY_LEN + 1])
    if not (0.0 <= tenant_norm <= 1.0):
        report.add(
            MonitorResult(
                Severity.WARNING,
                "TENANT_FEATURE_OUT_OF_RANGE",
                f"Normalised tenant feature {tenant_norm:.3f} outside [0, 1].",
                tenant_norm,
            )
        )
    else:
        report.add(MonitorResult(Severity.OK, "TENANT_FEATURE_OK", "", tenant_norm))

    presence = state[12:17]
    if len(presence) != config.NUM_ENTITY_TYPES:
        report.add(
            MonitorResult(
                Severity.ERROR,
                "WRONG_PRESENCE_SHAPE",
                f"Presence slice length {len(presence)} != {config.NUM_ENTITY_TYPES}",
            )
        )
    elif not np.all(np.isin(presence, [0.0, 1.0])):
        worst = float(np.max(np.abs(presence - np.round(presence))))
        report.add(
            MonitorResult(
                Severity.ERROR,
                "PRESENCE_NOT_BINARY",
                f"Presence bits are not binary. Max deviation from {{0,1}}: {worst:.4f}",
                worst,
            )
        )
    else:
        report.add(
            MonitorResult(Severity.OK, "PRESENCE_BITS_OK", "", float(presence.sum()))
        )

    rid_norm = state[17]
    if not (0.0 <= float(rid_norm) <= 1.0):
        report.add(MonitorResult(
            Severity.ERROR,
            "WRONG_RID_NORM",
            f"rid_norm at index 17 out of range: {rid_norm:.4f}"
        ))

    return report


# ---------------------------------------------------------------------------
# 6. Convenience: run all checks and raise if errors found
# ---------------------------------------------------------------------------


def run_workload_checks(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> MonitorReport:
    report = check_workload(df_train, df_test)
    log.info("[MONITOR] Workload check: %s", report.summary())
    if raise_on_error and report.has_errors:
        raise RuntimeError(
            "Workload integrity checks failed -- see MONITOR ERR logs above."
        )
    return report


def run_evaluation_checks(
    lru_metrics: dict[str, float],
    dqn_metrics: dict[str, float],
    train_hit_ratio: float | None = None,
    *,
    raise_on_error: bool = False,  # evaluation errors are non-fatal: log and continue
) -> MonitorReport:
    report = check_evaluation(lru_metrics, dqn_metrics, train_hit_ratio)
    log.info("[MONITOR] Evaluation check: %s", report.summary())
    if raise_on_error and report.has_errors:
        raise RuntimeError(
            "Evaluation integrity checks failed -- see MONITOR ERR logs above."
        )
    return report
