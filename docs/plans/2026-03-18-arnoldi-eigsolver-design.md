# GPU-friendly Arnoldi Eigensolver for VUMPS

## Problem

VUMPS uses `simple_eig` (power method) to solve three eigenvalue sub-problems (Eenv, ACenv, Cenv) per `leftmove` iteration. Power method converges as |lambda_1/lambda_2|^n, which is slow when the spectral gap is small. KrylovKit's `eigsolve` is the natural alternative but suffers from scalar indexing on GPU, making it impractical for large-scale problems.

Additionally, in iPEPS optimization, overly accurate eigensolves can hurt overall convergence or trap the optimizer in local minima. The goal is not maximum eigensolver accuracy, but a better accuracy-per-FLOP trade-off with controllable precision.

## Design

### Part 1: GPU-friendly Arnoldi Eigensolver

**Algorithm**: Standard Arnoldi iteration for the dominant eigenpair.

1. Given linear operator `f` and initial vector `v_0`, build k-dimensional Krylov subspace
2. Gram-Schmidt orthogonalization produces orthonormal basis `V = [v_1, ..., v_k]` and upper Hessenberg matrix `H` (k x k)
3. Solve small eigenproblem `H y = lambda y` on CPU (negligible cost for k ~ 5-20)
4. Reconstruct eigenvector in original space: `v = sum_i y_i * v_i`

**GPU compatibility**:
- Krylov vectors stored as `Vector{AbstractArray}` (not concatenated into a matrix)
- Orthogonalization uses `dot` + `axpy!` (BLAS-2, GPU-native)
- Hessenberg matrix `H` is k x k on CPU, eigendecomposition via `LinearAlgebra.eigen`
- Final reconstruction is a GPU linear combination

**Interface**:
```julia
function arnoldi_eig(f, v; krylov_dim=10, ifvalue=false)
    # Returns (lambda, v), same interface as simple_eig
end
```

**Integration with VUMPS**:
- New parameter `eigsolver::Symbol = :power` in VUMPS/QRCTM structs, options: `:power`, `:arnoldi`, `:krylovkit`
- New parameter `krylov_dim::Int = 10` (used when `eigsolver = :arnoldi`)
- New parameter `krylov_dim_ad::Int = 10` (for AD phase)
- `maxiter_power` retained for backward compatibility with `:power` mode
- `Cenv`, `Eenv`, `ACenv` dispatch to the chosen eigsolver

**AD compatibility**: Forward pass uses Arnoldi. Backward propagation uses existing implicit differentiation (Neumann series in `autodiff.jl`). No need to differentiate through Arnoldi itself.

### Part 2: Benchmark Framework

**Goal**: Systematically compare eigsolve strategies on iPEPS optimization quality and speed.

**Experiment grid**:

| Dimension       | Values                                              |
|-----------------|-----------------------------------------------------|
| Eigsolver       | `:power`, `:arnoldi`, `:krylovkit`                  |
| Precision param | power: maxiter_power in {1,2,5,10,20}; arnoldi: krylov_dim in {3,5,10,15,20} |
| Problem size    | At least one representative (D, chi) combination    |
| Metrics         | Per-step wall time, outer iterations, total wall time, final energy, energy vs time curve |

**Implementation**: `benchmark/eigsolver_benchmark.jl`

- Define parameter grid as array of configs
- For each config, run N steps of iPEPS optimization
- Record per-step: step index, energy, wall_time, vumps_iters, eigsolve_time
- Output to JLD2 for post-processing

**Instrumentation**:
- Add timing around eigsolve calls in `Cenv`, `Eenv`, `ACenv`
- Record VUMPS outer iteration `err` trajectory
- Record energy vs cumulative wall time

**Out of scope**:
- Automatic precision scheduling (study data first)
- Changes to the optimizer itself
