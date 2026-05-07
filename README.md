# mpGNE-ppopt: Multi-Agent Generalized Nash Equilibrium via Parametric Programming

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**mpGNE-ppopt** is a high-performance framework for solving Generalized Nash Equilibrium (GNE) problems in multi-agent systems using explicit multi-parametric Quadratic Programming (mpQP). It provides ultra-fast explicit solvers (FACET-GNE) and efficient iterative multiparametric methods (ImpGNE) that significantly outperform standard iterative methods like ADMM and Jacobi Best-Response.

## Key Features

- **FACET-GNE Solvers**: Explicit solvers using precomputed critical regions for ultra-fast online inference (< 1ms). Supports two neighbor-discovery strategies:
  - **FACET-H (Hyperplane Adjacency)**: Fast offline discovery based on shared hyperplanes.
  - **FACET-LP (Facet Adjacency)**: Exact (d-1)-dimensional facet discovery using LP (per ACC 2026).
- **ImpGNE Solver**: An iterative multiparametric Distributed GNE solver that leverages precomputed explicit solution maps to perform affine lookups instead of solving QPs during iterations.
- **High-Performance Iterative Baselines**: Integrated **OSQP**-based ADMM and Jacobi Best-Response (Projected Gradient) solvers for robust benchmarking. Supports fallback to SLSQP.
- **Multi-Agent Scaling**: Scalable implementation for $M \ge 4$ agents with support for coupling constraints and parallelized offline map generation.

## Performance Benchmark

In a typical 4-agent GNE scenario with coupling constraints and `osqp` backend:

| Method | Avg Time | Max Time | Avg Iters | Fallbacks |
| :--- | :--- | :--- | :--- | :--- |
| **Iterative (ADMM)** | ~3-5 ms | ~20 ms | 6-10 | — |
| **Iterative (Jacobi BR)** | ~120-200 ms | ~800 ms | 160-200 | — |
| **ImpGNE (Iterative mpQP)**| ~1.5-3 ms | ~10 ms | 120-150 | — |
| **FACET-H (Hyperplane)** | **~0.2 ms** | **~2 ms** | **N/A** | 0 |
| **FACET-LP (Facet LP)** | **~0.2 ms** | **~2 ms** | **N/A** | 0 |

**Speedup**: FACET-GNE variants achieve **~20x** speedup over ADMM, **~10x** over ImpGNE, and **>1000x** speedup over standard Jacobi Best-Response.

## Installation

```bash
git clone https://github.com/Parth-B7999/mpGNE-ppopt.git
cd mpGNE-ppopt
pip install -r requirements.txt
pip install osqp  # Recommended for performance
```

## Quick Start

Run the single-simulation demo to visualize trajectories and performance benchmarks:

```bash
python demo_single_run.py
```

This script will:
1. Generate a random multi-agent system.
2. Build the GNE game formulation.
3. Solve or load the offline mpQP critical regions.
4. Precompute **Hyperplane** and **Facet (LP)** neighbor maps.
5. Run a closed-loop simulation comparing **FACET-H**, **FACET-LP**, **ImpGNE**, **ADMM**, and **Jacobi BR**.
6. Generate a comprehensive performance report (`demo_benchmark.png`) and trajectory plots (`demo_trajectory.png`).

## Neighbor Adjacency Methods

- **Hyperplane Adjacency (FACET-H)**: Declares two regions as neighbors if they share any common supporting hyperplane. It is fast to compute offline but can be over-inclusive.
- **Facet Adjacency (FACET-LP)**: Rigorously confirms if the intersection of two regions is (d-1)-dimensional using a Linear Program. This is the method proposed in Saini et al. (ACC 2026) for exact, minimal neighbor sets.

## Repository Structure

- `mpgne/`: Core library containing solvers, game builders, and plant generators.
  - `facet_gne.py`: Online search and neighbor discovery logic.
  - `impdimc_solver.py`: ImpGNE iterative multiparametric solver.
- `tests/`: Comprehensive test suite for verification.
- `checkpoints_demo/`: Local storage for precomputed offline maps and neighbor lists.
- `demo_single_run.py`: Primary demonstration and benchmarking script.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
