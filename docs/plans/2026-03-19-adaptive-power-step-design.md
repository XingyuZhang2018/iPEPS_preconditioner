# Adaptive Power Step for VUMPS Forward Phase

## Date: 2026-03-19

## Motivation

In VUMPS iteration, the `maxiter_power` parameter controls how many power method steps are used per eigsolve call. Currently this is fixed throughout the iteration. The insight is:

- **Early iterations** (large err): environment is far from converged, expensive eigsolve is wasted
- **Late iterations** (small err): environment is nearly converged, more accurate eigsolve helps convergence

A fixed value forces a trade-off: too few steps wastes outer iterations near convergence, too many steps wastes compute early on.

## Design: Doubling Strategy (Method C)

### Algorithm

Record `err_init` from the first VUMPS step. At each subsequent step, compute how many orders of magnitude err has decreased:

```
n_decades = floor(Int, log10(err_init / err))
power_step = clamp(k_min * 2^n_decades, k_min, k_max)
```

Where:
- `k_min = 1` (always start with 1 power step)
- `k_max` = user-specified `maxiter_power` (acts as upper bound)
- `err_init` = automatically recorded from first iteration

### Example (k_max=10)

| err reduction | n_decades | power_step |
|---|---|---|
| < 10x | 0 | 1 |
| 10-100x | 1 | 2 |
| 100-1000x | 2 | 4 |
| 1000-10000x | 3 | 8 |
| > 10000x | 4+ | 10 (capped) |

### Interface

```julia
VUMPS(eigsolver=:power, maxiter_power=10, adaptive_power=true)
```

- `adaptive_power=false` (default): fixed power step, backward compatible
- `adaptive_power=true`: doubling schedule, `maxiter_power` becomes the ceiling

### Scope

- **Forward phase only**: adaptive scheduling applies in the `@ignore_derivatives` loop
- **AD phase unchanged**: uses fixed `maxiter_power_ad` as before

### Implementation Changes

1. **VUMPS struct** (`interface.jl`): add `adaptive_power::Bool = false`
2. **runtime.jl** forward loop: when `adaptive_power=true`, dynamically set `alg.maxiter_power` each step based on err
3. **Benchmark**: compare `power_k=5_fixed` vs `power_adaptive_max5` vs `power_adaptive_max10`

### Benchmark Plan

- D=3, chi=64, 50 optim steps, local preconditioner
- Configs: power_k=5 (fixed baseline), adaptive_max5, adaptive_max10
- Measure: energy trajectory, wall time, VUMPS iterations per optim step

## Context: Prior Experiments

From earlier benchmarking on this branch:
- `pow5_ad4 + local precondition` was found to be the optimal fixed configuration
- Momentum and Chebyshev acceleration of power method failed (operator changes each VUMPS step, history is harmful)
- Anderson mixing on MPS tensors is mathematically unsound (MPS not in linear space)
- Environment extrapolation similarly invalid

The adaptive power step approach avoids these issues: it only changes *how many* power iterations to run, not the iteration itself.
