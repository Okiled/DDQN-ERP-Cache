"""
Generate the synthetic ERP workload CSV.

The workload follows the TPC-C-inspired Pareto and sequential access pattern
described in Paper Section III.D and SPEC.md Section 4.
"""

# ruff: noqa: E402

import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from config import (
    NUM_TENANTS,
    SEQUENTIAL_TRANSITIONS,
    TENANT_PARETO_ALPHA,
    TEST_PARETO_WEIGHTS,
    TRAIN_PARETO_WEIGHTS,
    TRAIN_SIZE,
)

log = logging.getLogger(__name__)


def _build_tenant_weights(n_tenants: int, alpha: float) -> np.ndarray:
    """Compute normalised Pareto (power-law) weights for n_tenants."""
    ranks = np.arange(1, n_tenants + 1, dtype=float)
    weights = 1.0 / (ranks**alpha)
    normalised: np.ndarray = weights / float(weights.sum())
    return normalised


def _sample_entity(weights: list[float]) -> int:
    """Sample one entity type according to the entity-level Pareto weights."""
    entity_values = np.arange(len(weights))
    return int(np.random.choice(entity_values, p=weights))


def generate_workload(
    total_queries: int,
    seed: int,
    output_path: str,
    transitions: dict | None = None,
    pareto_alpha: float | None = None,
    entity_weights: list[float] | None = None,
) -> pd.DataFrame:
    """Generate workload rows and save them to output_path.

    Args:
        total_queries: Total number of queries to generate.
        seed: Random seed for reproducibility.
        output_path: CSV output path.
        transitions: Markov transition dict. Defaults to config.SEQUENTIAL_TRANSITIONS.
            Used by robustness sweep to simulate different ERP module behaviors.
        pareto_alpha: Tenant Pareto exponent. Defaults to config.TENANT_PARETO_ALPHA.
            Used by robustness sweep to simulate different tenant concentration levels.
        entity_weights: Entity-type sampling weights (must sum to 1, length = NUM_ENTITY_TYPES).
            When set, overrides both TRAIN_PARETO_WEIGHTS and TEST_PARETO_WEIGHTS for
            all queries. Used by real_erp robustness config to apply production-derived
            entity proportions (Odoo 14, 7,260 records, 2024-04 to 2026-06).
    """
    effective_transitions = transitions if transitions is not None else SEQUENTIAL_TRANSITIONS
    effective_alpha = pareto_alpha if pareto_alpha is not None else TENANT_PARETO_ALPHA

    log.info("[GENERATOR] Generating %d queries with seed %d...", total_queries, seed)
    random.seed(seed)
    np.random.seed(seed)

    tenant_pool = list(range(1, NUM_TENANTS + 1))
    tenant_weights = _build_tenant_weights(NUM_TENANTS, effective_alpha)

    entity_types: list[int] = []
    tenant_ids: list[int] = []
    sequential_count = 0
    tenant_last_entity = {}

    for i in range(total_queries):
        is_train = i < TRAIN_SIZE
        if entity_weights is not None:
            current_weights = entity_weights
        else:
            current_weights = TRAIN_PARETO_WEIGHTS if is_train else TEST_PARETO_WEIGHTS

        tenant_id = int(np.random.choice(tenant_pool, p=tenant_weights))
        last_entity = tenant_last_entity.get(tenant_id, -1)

        entity_type: int | None = None
        if last_entity in effective_transitions:
            targets, prob = effective_transitions[last_entity]
            if random.random() < prob:
                entity_type = int(random.choice(targets))
                sequential_count += 1

        if entity_type is None:
            entity_type = _sample_entity(current_weights)

        tenant_last_entity[tenant_id] = entity_type
        entity_types.append(entity_type)
        tenant_ids.append(tenant_id)

    record_ids = []
    rng = np.random.default_rng(seed + 1)
    for _ in range(total_queries):
        raw = rng.pareto(config.RECORD_PARETO_ALPHA)
        rid = int(raw % config.NUM_RECORDS_PER_ENTITY) + 1
        record_ids.append(rid)

    tid_dtype = np.int8 if NUM_TENANTS <= 127 else np.int16
    workload = pd.DataFrame(
        {
            "query_id": np.arange(total_queries, dtype=np.int64),
            "entity_type": np.array(entity_types, dtype=np.int8),
            "tenant_id": np.array(tenant_ids, dtype=tid_dtype),
            "record_id": record_ids,
        }
    )

    products_count = int((workload["entity_type"] == 0).sum())
    sales_count = int((workload["entity_type"] == 1).sum())
    invoices_count = int((workload["entity_type"] == 2).sum())
    company_count = int((workload["entity_type"] == 3).sum())
    assets_count = int((workload["entity_type"] == 4).sum())

    total_transitions = max(total_queries - 1, 0)
    sequential_pct = (
        (sequential_count / total_transitions) * 100.0 if total_transitions > 0 else 0.0
    )

    tenant_counts = workload["tenant_id"].value_counts(normalize=True)
    top2_pct = tenant_counts.iloc[:2].sum() * 100
    n_hot = max(1, round(NUM_TENANTS * 0.2))
    tophot_pct = tenant_counts.iloc[:n_hot].sum() * 100

    log.info(
        "[GENERATOR] Entity distribution: "
        "Products=%d, Sales=%d, Invoices=%d, Company=%d, Assets=%d",
        products_count, sales_count, invoices_count, company_count, assets_count
    )
    log.info(
        "[GENERATOR] Sequential transitions: %d / %d (%.1f%%)",
        sequential_count, total_transitions, sequential_pct,
    )
    log.info(
        "[GENERATOR] Tenant Pareto (alpha=%.1f, n=%d): "
        "top-2=%.1f%% top-%d (20%% of tenants)=%.1f%%",
        effective_alpha, NUM_TENANTS, top2_pct, n_hot, tophot_pct,
    )

    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        workload.to_csv(path, index=False)
    except OSError as exc:
        log.error("[ERROR] File write failed: %s", exc)
        sys.exit(1)

    log.info("[GENERATOR] Saved to %s", output_path)
    return workload
