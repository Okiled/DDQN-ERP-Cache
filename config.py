"""
Central constants for the ERP cache DQN simulation.

Values are sourced from the ICIMTech 2026 paper and AGENTS.md.
"""

# --- Hyperparameters (Paper Table I) ---
LEARNING_RATE = 0.001
GAMMA = 0.95
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.01

# --- Network Architecture (Paper Fig. 2) ---
# State vector = 10 query-history features + occupancy + tenant + 5 cache-presence bits
# + rid_norm (index 17: last record ID normalised, kept for model integrity).
STATE_DIM = 18
ACTION_DIM = 11
HIDDEN_DIM = 64

# --- Environment ---
MAX_CACHE_SIZE = 20
SIMULATED_REDIS_BASE_MS = 0.1
SIMULATED_PG_BASE_MS = 15.0
REWARD_HIT = 1.0
REWARD_MISS = -1.0

# --- Workload (Paper Section III.D) ---
TOTAL_QUERIES = 125_000
TRAIN_SIZE = 100_000
TEST_SIZE = 25_000

# Entity-type Pareto weights [products=0, sales_order=1, invoices=2, company=3, assets=4]
TRAIN_PARETO_WEIGHTS = [0.40, 0.30, 0.15, 0.10, 0.05]
TEST_PARETO_WEIGHTS = [0.36, 0.24, 0.26, 0.10, 0.04]

# Tenant Pareto power-law exponent for 10 tenants.
TENANT_PARETO_ALPHA = 2.0

SEQUENTIAL_PROB = 0.70

# Multi-step Markov chain transitions for supply chain business logic.
SEQUENTIAL_TRANSITIONS: dict[int, tuple[list[int], float]] = {
    1: ([0, 2], 0.70),
    0: ([1, 2], 0.40),
    2: ([3],    0.50),
}

# --- Multi-seed statistical experiment (R1+R2: generalizability) ---
EXPERIMENT_SEEDS = list(range(1, 21))
SIGNIFICANCE_ALPHA = 0.05

# --- Robustness sweep configurations (R1: overfitting, R2: generalizability/transfer) ---
# Different Markov configs represent different ERP module behaviors:
#   low_seq  = HR/Admin module (less sequential, more random access)
#   mid_seq  = Supply Chain module (current training config, moderately sequential)
#   high_seq = Manufacturing/Logistics module (highly sequential pipelines)
#   real_erp = Production Odoo 14 entity proportions (7,260 records, 2024-04 to 2026-06)
ROBUSTNESS_TRANSITIONS: dict[str, dict[int, tuple[list[int], float]]] = {
    "low_seq":  {1: ([0, 2], 0.50), 0: ([1, 2], 0.25), 2: ([3], 0.30)},
    "mid_seq":  {1: ([0, 2], 0.70), 0: ([1, 2], 0.40), 2: ([3], 0.50)},
    "high_seq": {1: ([0, 2], 0.90), 0: ([1, 2], 0.60), 2: ([3], 0.70)},
    "real_erp": {1: ([0, 2], 0.70), 0: ([1, 2], 0.40), 2: ([3], 0.50)},
}

ROBUSTNESS_MODULE_LABELS: dict[str, str] = {
    "low_seq":  "HR/Admin-like workload",
    "mid_seq":  "Supply Chain workload (training config)",
    "high_seq": "Manufacturing/Logistics-like workload",
    "real_erp": "Real ERP-Informed workload",
}

# Entity distribution from production Odoo 14 deployment.
# Source: 7,260 module activity records across 27 months (2024-04 to 2026-06).
# Mapping: Products=9%, Sales=31%, Invoices=18%, Company=26%, Assets=16%.
REAL_ERP_PARETO_WEIGHTS = [0.09, 0.31, 0.18, 0.26, 0.16]

# Different Pareto alphas simulate different tenant concentration levels:
#   1.5 = highly skewed (few dominant tenants, bursty)
#   2.0 = moderate skew (current training config)
#   2.5 = more uniform (many active tenants)
ROBUSTNESS_PARETO_ALPHAS = [1.5, 2.0, 2.5]
ROBUSTNESS_SEEDS = [42, 123, 777]

# --- Hyperparameter sensitivity (R2: sensitivity analysis, P2) ---
SENSITIVITY_LRS = [0.0001, 0.001, 0.01]

# --- Tenant config ---
NUM_TENANTS = 10

# --- Training ---
NUM_EPOCHS = 8
TRAIN_STEPS = 800_000
WARMUP_STEPS = 5_000
TARGET_UPDATE_FREQ = 10_000
TRAIN_FREQ = 4
EPSILON_DECAY = (EPSILON_START - EPSILON_END) * TRAIN_FREQ / (TRAIN_STEPS - WARMUP_STEPS)

# --- Named literals (no magic numbers in src/ files) ---
CACHE_TTL_SECONDS = 3_600
HISTORY_LEN = 10
ENTITY_NORMALIZER = 4.0
DUMMY_ROWS_PER_TABLE = 100
NUM_RECORDS_PER_ENTITY = 500
RECORD_PARETO_ALPHA = 1.5
ROLLING_MEAN_WINDOW = 50
CONVERGENCE_ROLLING_WINDOW = 500

# --- Derived constants (computed from above, not hardcoded) ---
ENTITY_TABLE_NAMES = ("products", "sales_orders", "invoices", "company", "assets")
NUM_ENTITY_TYPES = len(TRAIN_PARETO_WEIGHTS)

# --- Paths ---
WORKLOAD_PATH = "data/workload.csv"
MODEL_PATH = "models/dqn_agent.pth"
RESULTS_DIR = "results/"
