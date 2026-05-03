# mpGNE-ppopt: Multi-Agent Generalized Nash Equilibrium via Parametric Programming

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**mpGNE-ppopt** is a high-performance framework for solving Generalized Nash Equilibrium (GNE) problems in multi-agent systems using explicit multi-parametric Quadratic Programming (mpQP). It provides a sub-millisecond explicit solver (FACET-GNE) that significantly outperforms standard iterative methods like ADMM and Jacobi Best-Response.

## Key Features

- **FACET-GNE Solver**: An explicit solver that uses precomputed critical regions for ultra-fast online inference (< 1ms).
- **Multi-Agent Scaling**: Scalable implementation for $M \ge 4$ agents with support for coupling constraints.
- **Iterative Baselines**: Integrated ADMM and Jacobi Best-Response solvers for performance benchmarking.
- **Offline mpQP Generation**: Parallelized offline map generation using `ppopt`.
- **Robust Fallback**: Automatic ADMM-based fallback for state regions outside precomputed maps.

## Performance Benchmark

In a typical 4-agent DiMPC scenario with coupling constraints:

| Method | Avg Time | Max Time | Avg Iters |
| :--- | :--- | :--- | :--- |
| **Iterative (ADMM)** | ~3-5 ms | ~50 ms | 6-10 |
| **Iterative (Jacobi BR)** | ~120-200 ms | ~2000 ms | 160-200 |
| **Explicit (FACET-GNE)** | **~0.2 ms** | **~10 ms** | **N/A** |

**Speedup**: FACET-GNE achieves **~20x** speedup over ADMM and **>1000x** speedup over Jacobi Best-Response in high-coupling regimes.

## Installation

```bash
git clone https://github.com/Parth-B7999/mpGNE-ppopt.git
cd mpGNE-ppopt
pip install -r requirements.txt
```

## Quick Start

Run the single-simulation demo to visualize trajectory and performance:

```bash
python demo_single_run.py
```

This script will:
1. Generate a random multi-agent system.
2. Build the GNE game formulation.
3. Solve or load the offline mpQP critical regions.
4. Run a closed-loop simulation comparing FACET-GNE, ADMM, and Jacobi BR.
5. Generate a performance report and trajectory plots.

## Repository Structure

- `mpgne/`: Core library containing solvers, game builders, and plant generators.
- `tests/`: Comprehensive test suite for verification.
- `checkpoints_demo/`: (Local) Storage for precomputed offline maps.
- `demo_single_run.py`: Primary demonstration and benchmarking script.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{saini2025facet,
  title={FACET-GNE: Explicit Solutions for Multi-Agent Generalized Nash Equilibrium},
  author={Saini et al.},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
