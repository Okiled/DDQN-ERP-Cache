# Proactive Cache Optimization in Enterprise Systems Through Double Deep Reinforcement Learning

This repository contains the source code, environment simulations, and experimental data for the paper **"Proactive Cache Optimization in Enterprise Systems Through Double Deep Reinforcement Learning"**, accepted at **ICIMTech 2026**.

This repository is provided for **Data Availability** and reproducibility purposes.

## Overview

Cloud-hosted Enterprise Resource Planning (ERP) systems running on multi-tenant backends regularly hit latency problems once pushed past their load envelope. Conventional cache policies like Least Recently Used (LRU) act reactively and cannot predict sequential business-logic workflows (e.g., Sales Order → Inventory → Delivery).

This project introduces a **proactive caching middleware** powered by a Double Deep Q-Network (DDQN). The agent observes an 18-dimensional state vector to predict and prefetch the next necessary entity types before the user requests them, while simultaneously handling cache evictions.

### Experimental Baselines
The repository implements and benchmarks the following caching policies within the exact same sequence generator:
1. **LRU** (Least Recently Used)
2. **LFU** (Least Frequently Used)
3. **ARC** (Adaptive Replacement Cache)
4. **SLRU** (Segmented LRU)
5. **DDQN** (Proposed Proactive Agent)

### Key Results
- **93.30%** mean cache hit ratio across 20 random seeds on synthetic TPC-C supply-chain workloads.
- **40.47% reduction** in mean analytical query latency compared to standard LRU.
- Stable zero-shot deployment boundaries verified via an Odoo 14 real-world data set.

---

## Citation

If you find this repository or our paper useful in your research, please consider citing our work:

```bibtex
@inproceedings{tanuwijaya2026proactive,
  title={Proactive Cache Optimization in Enterprise Systems Through Double Deep Reinforcement Learning},
  author={Tanuwijaya Tariono, Jason and Hartono, Deliko and Setiawan, Reina},
  booktitle={2026 International Conference on Information Management and Technology (ICIMTech)},
  year={2026},
  organization={IEEE}
}
```

---

## Repository Structure

```text
├── data/                  # Transaction access traces and workload configurations
├── models/                # Saved DDQN PyTorch model checkpoints (.pt)
├── results/               # Experimental outputs, logs, and CSV comparison tables
├── src/                   # Source code for the DDQN agent, environment, and baselines
├── main.py                # Main execution script to run training & evaluations
├── monitor_server.py      # Lightweight monitoring script for VPS deployments
├── docker-compose.yml     # Docker configuration for Redis and PostgreSQL
├── requirements.txt       # Python dependencies
└── config.py              # Centralized hyperparameters and system configurations
```

---

## System Requirements

The simulation pipeline requires the following components:
- **Python 3.10+**
- **Docker & Docker Compose** (for running local Redis and PostgreSQL instances)
- At least 2GB of free RAM and 1GB of disk space.

### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## Running the Experiments

### 1. Start the Environment Backends
The environment simulates cache hits using Redis and cache misses using PostgreSQL. Start the containers using:
```bash
docker-compose up -d
```

### 2. Run the DDQN Agent
The main driver script automates the preflight checks, initializes the MDP environment, trains the agent, and benchmarks it against the standard heuristics (LRU, LFU, ARC, SLRU).
```bash
python main.py
```
*Note: A complete run across 20 seeds may take several hours depending on CPU capabilities. The system features an automatic checkpointing mechanism in `results/checkpoint.json` so that progress is saved after every seed.*

---

## Deployment and Resilience

If running on a headless VPS (e.g., Ubuntu 24.04 Droplet), the script automatically uses the `Agg` matplotlib backend to save plots directly to the `results/` folder. The built-in `monitor_server.py` can be used to track system resource utilization (CPU, RAM, DB connection limits) during the simulation.

### Automatic Recovery
Due to the long-running nature of the training sweeps, the system automatically writes to `results/checkpoint.json`. If the process is killed or the server restarts, simply re-run `python main.py` and the pipeline will skip already-completed seeds.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute the source code for both academic and commercial purposes, provided that proper credit is given to the original authors. 

The datasets and transaction traces in the `data/` directory are provided for research purposes under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.
