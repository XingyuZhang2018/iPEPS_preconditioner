# iPEPS_preconditioner

Accelerating two-dimensional tensor network optimization by preconditioning.

This repository contains Julia code supporting the methods described in:

Xing-Yu Zhang, Qi Yang, Philippe Corboz, Jutho Haegeman, Wei Tang (2025), "Accelerating two-dimensional tensor network optimization by preconditioning". Available at: https://arxiv.org/abs/2511.09546v1

Overview
--------

- Implements preconditioning strategies for variational optimization of infinite projected entangled pair states (iPEPS).
- Includes environment solvers, optimization drivers, and utilities for running benchmarks reported in the paper.
- Supports only single-unit-cell iPEPS with C4v symmetry. For simulations with larger unit cells, see https://github.com/XingyuZhang2018/TeneT_demo/tree/Array-of-Array

Contents
--------

- `src/` — core Julia implementation (optimization, contraction, preconditioners).
- `data/` — example results and checkpoints used for reproduction.
- `README.md` — this file.

Quickstart
----------

Prerequisites:

- Julia 1.11 (or current tested version documented in `Project.toml`).
- Add required packages by activating the project and running `] instantiate` in the Julia REPL.

Example workflow:

1. Activate the project in Julia:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

2. Run a small example using the provided optimization driver (see `src/ipeps_optimization/optimise_ipeps.jl`).

Example
-------

The following minimal example demonstrates calling `optimise_ipeps`. Replace placeholders with your model and initial tensors.

Example parameter sets and checkpointed runs are available under `data/`.

Citation
--------

If you use this code in your research, please cite the paper:

Zhang, X.-Y., Yang, Q., Corboz, P., Haegeman, J., & Tang, W. (2025). Accelerating two-dimensional tensor network optimization by preconditioning. arXiv:2511.09546v1. https://arxiv.org/abs/2511.09546v1

License
-------

This repository is distributed under the terms of the LICENSE file at the repository root.

Contact
-------

For questions about the implementation or reproducibility, open an issue or contact the authors listed in the paper.

[1]: https://arxiv.org/abs/2511.09546v1 "Accelerating two-dimensional tensor network optimization by preconditioning"


