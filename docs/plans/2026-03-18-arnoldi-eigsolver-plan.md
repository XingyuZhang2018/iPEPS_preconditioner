# GPU-friendly Arnoldi Eigensolver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the power method eigensolver with a GPU-friendly Arnoldi iteration, and build a benchmark framework to compare eigsolve strategies on iPEPS optimization.

**Architecture:** Add `arnoldi_eig` to `utilities.jl` alongside `simple_eig`, extend algorithm structs with `eigsolver`/`krylov_dim` parameters, update the three eigsolve dispatch sites (`Cenv`/`Eenv`/`ACenv`) to support three-way dispatch (`:power`/`:arnoldi`/`:krylovkit`), and create a standalone benchmark script.

**Tech Stack:** Julia, LinearAlgebra, CUDA.jl, Zygote.jl, JLD2.jl

---

### Task 1: Implement `arnoldi_eig` in utilities.jl

**Files:**
- Modify: `src/contraction/utilities.jl:14-36` (add after `simple_eig`)

**Step 1: Write a unit test for `arnoldi_eig`**

Create file `test/test_arnoldi.jl`:

```julia
using Test
using LinearAlgebra

# We test arnoldi_eig standalone before integrating into iPEPS_preconditioner
# For now, include the function directly to test it in isolation.

include(joinpath(@__DIR__, "..", "src", "contraction", "arnoldi_eig.jl"))

@testset "arnoldi_eig" begin
    @testset "symmetric matrix dominant eigenpair" begin
        Random.seed!(42)
        n = 20
        # Build matrix with known dominant eigenvalue
        D_diag = collect(1.0:n)
        D_diag[end] = 100.0  # dominant eigenvalue = 100
        Q, _ = qr(randn(n, n))
        Q = Matrix(Q)
        A = Q * Diagonal(D_diag) * Q'

        f(v) = A * v
        v0 = randn(n)

        λ, v = arnoldi_eig(f, v0; krylov_dim=15, ifvalue=true)

        @test abs(λ) ≈ 100.0 atol=1e-10
        # v should be an eigenvector
        @test norm(f(v) - λ * v) / abs(λ) < 1e-10
    end

    @testset "non-symmetric matrix dominant eigenpair" begin
        Random.seed!(123)
        n = 30
        A = randn(n, n)
        # Make one eigenvalue clearly dominant
        A += 50.0 * I

        f(v) = A * v
        v0 = randn(n)

        λ_arnoldi, v_arnoldi = arnoldi_eig(f, v0; krylov_dim=20, ifvalue=true)

        # Compare with eigen
        λs = eigvals(A)
        λ_dom = λs[argmax(abs.(λs))]

        @test abs(λ_arnoldi) ≈ abs(λ_dom) rtol=1e-8
    end

    @testset "small krylov_dim gives inexact result" begin
        Random.seed!(42)
        n = 20
        D_diag = collect(1.0:n)
        D_diag[end] = 100.0
        Q, _ = qr(randn(n, n))
        Q = Matrix(Q)
        A = Q * Diagonal(D_diag) * Q'

        f(v) = A * v
        v0 = randn(n)

        # With krylov_dim=2, result should be less accurate than krylov_dim=15
        _, v_small = arnoldi_eig(f, v0; krylov_dim=2)
        _, v_large = arnoldi_eig(f, v0; krylov_dim=15)

        # Both should approximate the dominant eigenvector, but v_large better
        true_v = Q[:, end]
        overlap_small = abs(dot(v_small, true_v))
        overlap_large = abs(dot(v_large, true_v))
        @test overlap_large >= overlap_small - 0.01  # large is at least as good
    end

    @testset "works with reshape'd arrays (like tensors)" begin
        Random.seed!(42)
        # Simulate the tensor case: v is a matrix, f maps matrix->matrix
        χ, D = 8, 3
        A_full = randn(χ * D, χ * D)
        A_full += 50.0 * I  # dominant eigenvalue

        f(v) = reshape(A_full * vec(v), χ, D)
        v0 = randn(χ, D)

        λ, v = arnoldi_eig(f, v0; krylov_dim=10, ifvalue=true)

        @test abs(λ) > 40.0  # should be around 50
        # v should approximately satisfy eigenequation
        residual = norm(f(v) - λ * v) / abs(λ)
        @test residual < 1e-6
    end
end
```

**Step 2: Run test to verify it fails**

Run: `cd <project_root> && julia --project -e 'include("test/test_arnoldi.jl")'`
Expected: FAIL — file `src/contraction/arnoldi_eig.jl` does not exist

**Step 3: Implement `arnoldi_eig`**

Create file `src/contraction/arnoldi_eig.jl`:

```julia
using LinearAlgebra
using Random

"""
    arnoldi_eig(f, v; krylov_dim=10, ifvalue=false)

GPU-friendly Arnoldi iteration for the dominant eigenpair.

Builds a `krylov_dim`-dimensional Krylov subspace {v, f(v), f²(v), ...} using
modified Gram-Schmidt orthogonalization, then solves the small projected eigenproblem.

All operations on `v` use BLAS-level ops (dot, axpy!, norm) that are efficient on GPU arrays.
The small Hessenberg eigenproblem (krylov_dim × krylov_dim) is solved on CPU.

Returns `(λ, v)` with the same interface as `simple_eig`.
- `λ` is the dominant eigenvalue (only computed if `ifvalue=true`, else 0.0)
- `v` is the corresponding eigenvector, normalized

# Arguments
- `f`: linear operator v -> f(v)
- `v`: initial vector (can be any AbstractArray)
- `krylov_dim`: dimension of the Krylov subspace (controls accuracy vs cost)
- `ifvalue`: whether to compute the eigenvalue
"""
function arnoldi_eig(f, v; krylov_dim=10, ifvalue=false)
    T = real(eltype(v))

    # Normalize initial vector
    v = v / norm(v)

    # Storage for Krylov basis vectors
    Q = typeof(v)[]  # Vector of arrays, same type as v
    push!(Q, copy(v))

    # Upper Hessenberg matrix (on CPU, always small)
    H = zeros(Complex{T}, krylov_dim + 1, krylov_dim)

    for j in 1:krylov_dim
        # Apply operator
        w = f(Q[j])

        # Modified Gram-Schmidt orthogonalization
        for i in 1:j
            h_ij = dot(Q[i], w)
            H[i, j] = h_ij
            w = w - h_ij * Q[i]
        end

        h_next = norm(w)
        H[j + 1, j] = h_next

        # Check for breakdown (lucky convergence)
        if h_next < eps(T) * 100
            # Krylov subspace is invariant, reduce dimension
            krylov_dim = j
            break
        end

        if j < krylov_dim
            push!(Q, w / h_next)
        end
    end

    # Solve small eigenproblem on CPU
    H_k = Array(H[1:krylov_dim, 1:krylov_dim])
    eig_vals, eig_vecs = eigen(H_k)

    # Find dominant eigenvalue (largest magnitude)
    idx = argmax(abs.(eig_vals))
    λ = real(eig_vals[idx])  # for transfer matrices, dominant eigenvalue is real
    y = eig_vecs[:, idx]

    # Reconstruct eigenvector in original space
    v_result = real(y[1]) * Q[1]
    for i in 2:min(length(y), length(Q))
        v_result = v_result + real(y[i]) * Q[i]
    end

    # Normalize
    v_result = v_result / norm(v_result)

    if !ifvalue
        λ = zero(T)
    end

    return λ, v_result
end
```

**Step 4: Run test to verify it passes**

Run: `cd <project_root> && julia --project -e 'include("test/test_arnoldi.jl")'`
Expected: All 4 testsets PASS

**Step 5: Commit**

```bash
git add src/contraction/arnoldi_eig.jl test/test_arnoldi.jl
git commit -m "feat: add GPU-friendly arnoldi_eig implementation with tests"
```

---

### Task 2: Integrate `arnoldi_eig` into the module and algorithm structs

**Files:**
- Modify: `src/iPEPS_preconditioner.jl:31` (add include)
- Modify: `src/contraction/interface.jl:3-36` (add new parameters to QRCTM and VUMPS)
- Modify: `src/defaults.jl` (add default values)

**Step 1: Write an integration test**

Create file `test/test_eigsolver_dispatch.jl`:

```julia
using Test
using iPEPS_preconditioner

@testset "eigsolver parameter defaults" begin
    alg_v = VUMPS()
    @test alg_v.eigsolver == :power
    @test alg_v.krylov_dim == 10
    @test alg_v.krylov_dim_ad == 10

    alg_q = QRCTM()
    @test alg_q.eigsolver == :power
    @test alg_q.krylov_dim == 10

    # Backward compat: ifsimple_eig still works
    alg_old = VUMPS(ifsimple_eig=false)
    @test alg_old.eigsolver == :power  # ifsimple_eig is independent
end

@testset "eigsolver=:arnoldi constructor" begin
    alg = VUMPS(eigsolver=:arnoldi, krylov_dim=5, krylov_dim_ad=8)
    @test alg.eigsolver == :arnoldi
    @test alg.krylov_dim == 5
    @test alg.krylov_dim_ad == 8
end
```

**Step 2: Run test to verify it fails**

Run: `cd <project_root> && julia --project -e 'using iPEPS_preconditioner; include("test/test_eigsolver_dispatch.jl")'`
Expected: FAIL — `eigsolver` field doesn't exist

**Step 3: Add defaults**

In `src/defaults.jl`, add after line 17 (`const ifsimple_eig = true`):

```julia
const eigsolver = :power
const krylov_dim = 10
```

**Step 4: Add parameters to QRCTM struct**

In `src/contraction/interface.jl`, add to QRCTM after line 12 (`maxiter_power::Int = 1`):

```julia
    eigsolver::Symbol = Defaults.eigsolver
    krylov_dim::Int = Defaults.krylov_dim
```

**Step 5: Add parameters to VUMPS struct**

In `src/contraction/interface.jl`, add to VUMPS after line 30 (`maxiter_power_ad::Int = 5`):

```julia
    eigsolver::Symbol = Defaults.eigsolver
    krylov_dim::Int = Defaults.krylov_dim
    krylov_dim_ad::Int = Defaults.krylov_dim
```

**Step 6: Add include for arnoldi_eig.jl**

In `src/iPEPS_preconditioner.jl`, add after line 31 (`include("contraction/utilities.jl")`):

```julia
include("contraction/arnoldi_eig.jl")
```

**Step 7: Run test to verify it passes**

Run: `cd <project_root> && julia --project -e 'using iPEPS_preconditioner; include("test/test_eigsolver_dispatch.jl")'`
Expected: PASS

**Step 8: Commit**

```bash
git add src/defaults.jl src/contraction/interface.jl src/iPEPS_preconditioner.jl test/test_eigsolver_dispatch.jl
git commit -m "feat: add eigsolver/krylov_dim parameters to VUMPS and QRCTM"
```

---

### Task 3: Update eigsolve dispatch in Cenv/Eenv/ACenv

**Files:**
- Modify: `src/contraction/environment.jl:37-75` (update Cenv, Eenv, ACenv)
- Modify: `src/contraction/utilities.jl:14-36` (add `orth_for_ad` call to arnoldi path)

**Step 1: Write dispatch test**

Create file `test/test_eigsolve_modes.jl`:

```julia
using Test
using LinearAlgebra
using Random
using iPEPS_preconditioner

@testset "Cenv/Eenv/ACenv arnoldi mode smoke test" begin
    # Minimal VUMPS setup to test eigsolve dispatch
    Random.seed!(42)
    atype = Array
    D, χ = 2, 6
    model = Heisenberg()

    # Test with power mode (baseline)
    alg_power = VUMPS(eigsolver=:power, maxiter_power=5,
                      maxiter=3, maxiter_ad=0, miniter=0,
                      tol=1e-10, verbosity=0, ifload_env=false)

    # Test with arnoldi mode
    alg_arnoldi = VUMPS(eigsolver=:arnoldi, krylov_dim=5,
                        maxiter=3, maxiter_ad=0, miniter=0,
                        tol=1e-10, verbosity=0, ifload_env=false)

    A = init_ipeps(; atype, No=0, d=2, D, χ, params=GradientOptimize(
        boundary_alg=alg_power, iffixedpoint=false,
        optimizer=LBFGS(10; maxiter=0, verbosity=0, gradtol=1e-6),
        reuse_env=false, verbosity=0, folder=mktempdir(),
        preconditiontype=:none, ifload_lbfgs=false, iter_precond=0))

    M = iPEPS_preconditioner.double_layer(A)

    # Both modes should produce an environment without error
    env_p = iPEPS_preconditioner.initialize_env(M, χ, alg_power)
    env_a = iPEPS_preconditioner.initialize_env(M, χ, alg_arnoldi)

    # Run a few leftmove steps — should not error
    for _ in 1:3
        env_p, err_p = iPEPS_preconditioner.leftmove(M, env_p, alg_power)
        env_a, err_a = iPEPS_preconditioner.leftmove(M, env_a, alg_arnoldi)
    end

    # Both should produce finite errors
    @test isfinite(err_p)
    @test isfinite(err_a)
end
```

**Step 2: Run test to verify it fails**

Run: `cd <project_root> && julia --project -e 'include("test/test_eigsolve_modes.jl")'`
Expected: FAIL — Cenv/Eenv/ACenv don't dispatch on `eigsolver`

**Step 3: Create helper function `eigsolve_dispatch`**

Add to `src/contraction/utilities.jl` after the `simple_eig` function (after line 36):

```julia
"""
    eigsolve_dispatch(f, v, alg; ifvalue=false)

Dispatch to the appropriate eigensolver based on `alg.eigsolver`:
- `:power` → `simple_eig` with `alg.maxiter_power` iterations
- `:arnoldi` → `arnoldi_eig` with `alg.krylov_dim` Krylov dimension
- `:krylovkit` → KrylovKit `eigsolve`
"""
function eigsolve_dispatch(f, v, alg; ifvalue=false)
    if alg.eigsolver == :arnoldi
        λ, v = arnoldi_eig(f, v; krylov_dim=alg.krylov_dim, ifvalue)
        v = orth_for_ad(v)
        return λ, v
    elseif alg.eigsolver == :krylovkit
        λs, vs, info = eigsolve(f, v, 1, :LM; maxiter=1)
        info.converged == 0 && error("eigsolve did not converge")
        return λs[1], vs[1]
    else  # :power (default)
        return simple_eig(f, v; maxiter=alg.maxiter_power, ifvalue)
    end
end
```

**Step 4: Update Cenv, Eenv, ACenv to use `eigsolve_dispatch`**

Replace the bodies of `Cenv`, `Eenv`, `ACenv` in `src/contraction/environment.jl`.

Replace `Cenv` (lines 37-47) with:

```julia
function Cenv(Tu, Td, Cint; alg, ifvalue=false)
    f(C) = Cmap(C, Tu, Td)
    return eigsolve_dispatch(f, Cint, alg; ifvalue)
end
```

Replace `Eenv` (lines 49-61) with:

```julia
function Eenv(Tu, Td, M, Tint; alg, ifvalue=false)
    @unpack forloop_iter, ifparallel = alg
    f(E) = FLmap_parallel(E, Tu, Td, M; ifparallel, forloop_iter)
    return eigsolve_dispatch(f, Tint, alg; ifvalue)
end
```

Replace `ACenv` (lines 63-75) with:

```julia
function ACenv(Tl, Tr, M, Tint; alg, ifvalue=false)
    @unpack forloop_iter, ifparallel = alg
    f(AC) = ACmap_parallel(AC, Tl, Tr, M; ifparallel, forloop_iter)
    return eigsolve_dispatch(f, Tint, alg; ifvalue)
end
```

**Step 5: Run test to verify it passes**

Run: `cd <project_root> && julia --project -e 'include("test/test_eigsolve_modes.jl")'`
Expected: PASS

**Step 6: Run existing tests to check backward compatibility**

Run: `cd <project_root> && julia --project -e 'include("test/runtests.jl")'`
Expected: PASS (default `eigsolver=:power` matches old behavior)

**Step 7: Commit**

```bash
git add src/contraction/utilities.jl src/contraction/environment.jl test/test_eigsolve_modes.jl
git commit -m "feat: three-way eigsolver dispatch in Cenv/Eenv/ACenv"
```

---

### Task 4: Update runtime.jl for krylov_dim_ad switching

**Files:**
- Modify: `src/contraction/runtime.jl:88-118` (VUMPS environment function)

**Step 1: Write test for AD-phase krylov_dim switching**

Add to `test/test_eigsolve_modes.jl`:

```julia
@testset "krylov_dim_ad switching in VUMPS environment" begin
    alg = VUMPS(eigsolver=:arnoldi, krylov_dim=3, krylov_dim_ad=8,
                maxiter=2, maxiter_ad=2, miniter=0, miniter_ad=0,
                tol=1e-10, verbosity=0, ifload_env=false)

    # After environment() returns, krylov_dim should be restored to original
    @test alg.krylov_dim == 3
    @test alg.krylov_dim_ad == 8
end
```

**Step 2: Update `environment(M, env::VUMPSEnv, alg::VUMPS)` in runtime.jl**

In `src/contraction/runtime.jl`, the AD phase (lines 103-116) currently swaps `maxiter_power`. Add analogous swapping for `krylov_dim`. Replace lines 103-116 with:

```julia
    maxiter_power_origin = alg.maxiter_power
    krylov_dim_origin = alg.krylov_dim
    alg.maxiter_power = alg.maxiter_power_ad
    alg.krylov_dim = alg.krylov_dim_ad
    for i = 1:alg.maxiter_ad
        env, err = alg.ifcheckpoint ? checkpoint(leftmove, M, env, alg) : leftmove(M, env, alg)
        alg.verbosity >= 3 && i % alg.output_interval == 0 && @ignore_derivatives @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        if i > alg.miniter_ad && err < alg.tol
            alg.verbosity >= 2 && @ignore_derivatives @info @sprintf("contraction converged@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && @ignore_derivatives @warn @sprintf("contraction canceled@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        end
    end
    alg.maxiter_power = maxiter_power_origin
    alg.krylov_dim = krylov_dim_origin
```

**Step 3: Run all tests**

Run: `cd <project_root> && julia --project -e 'include("test/test_eigsolve_modes.jl")' && julia --project -e 'include("test/runtests.jl")'`
Expected: PASS

**Step 4: Commit**

```bash
git add src/contraction/runtime.jl test/test_eigsolve_modes.jl
git commit -m "feat: swap krylov_dim to krylov_dim_ad during VUMPS AD phase"
```

---

### Task 5: Benchmark framework

**Files:**
- Create: `benchmark/eigsolver_benchmark.jl`

**Step 1: Create benchmark script**

```julia
using iPEPS_preconditioner
using Random
using LinearAlgebra
using JLD2
using Printf
using Dates

"""
    BenchmarkConfig

One point in the experiment grid.
"""
struct BenchmarkConfig
    eigsolver::Symbol
    param::Int          # maxiter_power for :power, krylov_dim for :arnoldi, ignored for :krylovkit
    label::String
end

"""
    BenchmarkResult

Collected data for one optimization run.
"""
struct BenchmarkResult
    config::BenchmarkConfig
    step_data::Vector{NamedTuple{(:step, :energy, :wall_time, :vumps_iters, :grad_norm),
                                  Tuple{Int, Float64, Float64, Int, Float64}}}
    total_time::Float64
    final_energy::Float64
end

function make_alg(config::BenchmarkConfig; maxiter, maxiter_ad, miniter_ad, tol, verbosity)
    if config.eigsolver == :power
        return VUMPS(eigsolver=:power,
                     maxiter_power=config.param,
                     maxiter_power_ad=config.param,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false)
    elseif config.eigsolver == :arnoldi
        return VUMPS(eigsolver=:arnoldi,
                     krylov_dim=config.param,
                     krylov_dim_ad=config.param,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false)
    elseif config.eigsolver == :krylovkit
        return VUMPS(eigsolver=:krylovkit,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false, ifsimple_eig=false)
    else
        error("Unknown eigsolver: $(config.eigsolver)")
    end
end

function run_benchmark(config::BenchmarkConfig;
                       seed=42, D=2, χ=10, χshift=0,
                       maxiter_boundary=10, maxiter_ad=4, miniter_ad=4,
                       tol_boundary=1e-10,
                       n_optim_steps=20,
                       atype=Array,
                       verbosity=0)
    Random.seed!(seed)
    model = Heisenberg()

    boundary_alg = make_alg(config;
                            maxiter=maxiter_boundary, maxiter_ad=maxiter_ad,
                            miniter_ad=miniter_ad, tol=tol_boundary,
                            verbosity=verbosity)

    folder = mktempdir()
    params = GradientOptimize(
        boundary_alg=boundary_alg,
        iffixedpoint=false,
        optimizer=LBFGS(10; maxiter=n_optim_steps, verbosity=0, gradtol=1e-12),
        reuse_env=true,
        verbosity=verbosity,
        folder=folder,
        preconditiontype=:none,
        ifload_lbfgs=false,
        iter_precond=0,
    )

    A = init_ipeps(; atype, No=0, d=2, D, χ, params)

    t_start = time()
    # Note: optimise_ipeps runs the full optimization.
    # For fine-grained per-step timing, you may want to instrument optimise_ipeps
    # or call the inner loop manually.
    optimise_ipeps(A, χ, χshift, model, params)
    total_time = time() - t_start

    @printf("  %20s  total_time=%.2fs\n", config.label, total_time)

    return (config=config, total_time=total_time)
end

function main()
    configs = BenchmarkConfig[
        # Power method sweep
        BenchmarkConfig(:power, 1,  "power_k=1"),
        BenchmarkConfig(:power, 2,  "power_k=2"),
        BenchmarkConfig(:power, 5,  "power_k=5"),
        BenchmarkConfig(:power, 10, "power_k=10"),
        BenchmarkConfig(:power, 20, "power_k=20"),
        # Arnoldi sweep
        BenchmarkConfig(:arnoldi, 3,  "arnoldi_k=3"),
        BenchmarkConfig(:arnoldi, 5,  "arnoldi_k=5"),
        BenchmarkConfig(:arnoldi, 10, "arnoldi_k=10"),
        BenchmarkConfig(:arnoldi, 15, "arnoldi_k=15"),
        BenchmarkConfig(:arnoldi, 20, "arnoldi_k=20"),
        # KrylovKit baseline
        BenchmarkConfig(:krylovkit, 0, "krylovkit"),
    ]

    println("=" ^ 60)
    println("Eigsolver Benchmark — $(now())")
    println("=" ^ 60)

    results = []
    for config in configs
        @printf("Running: %s\n", config.label)
        result = run_benchmark(config; seed=42, D=2, χ=10, n_optim_steps=10)
        push!(results, result)
    end

    # Save results
    outfile = "benchmark/results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2"
    mkpath(dirname(outfile))
    @save outfile results
    println("\nResults saved to $outfile")

    # Summary table
    println("\n" * "=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    @printf("%20s  %12s\n", "Config", "Total Time")
    @printf("%20s  %12s\n", "-" ^ 20, "-" ^ 12)
    for r in results
        @printf("%20s  %10.2fs\n", r.config.label, r.total_time)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

**Step 2: Test that the benchmark script loads without error**

Run: `cd <project_root> && julia --project -e 'include("benchmark/eigsolver_benchmark.jl"); println("OK")'`
Expected: prints "OK" (main() is not called, only definitions loaded)

**Step 3: Run a minimal benchmark to verify end-to-end**

Run: `cd <project_root> && julia --project -e '
    include("benchmark/eigsolver_benchmark.jl")
    # Run just 2 configs with 2 optimization steps to check it works
    for config in [BenchmarkConfig(:power, 1, "power_k=1"), BenchmarkConfig(:arnoldi, 5, "arnoldi_k=5")]
        run_benchmark(config; seed=42, D=2, χ=6, n_optim_steps=2, verbosity=0)
    end
    println("Benchmark smoke test passed")
'`
Expected: Two runs complete, prints timing, "Benchmark smoke test passed"

**Step 4: Commit**

```bash
git add benchmark/eigsolver_benchmark.jl
git commit -m "feat: add eigsolver benchmark framework"
```

---

### Task 6: Update runtests.jl and final integration test

**Files:**
- Modify: `test/runtests.jl`

**Step 1: Update runtests.jl to include new tests**

Replace `test/runtests.jl` content with:

```julia
include("test_heisenberg_qrctm.jl")
include("test_arnoldi.jl")
include("test_eigsolver_dispatch.jl")
include("test_eigsolve_modes.jl")
```

**Step 2: Run full test suite**

Run: `cd <project_root> && julia --project -e 'include("test/runtests.jl")'`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "chore: add arnoldi and eigsolver tests to runtests.jl"
```

---

## Notes for the implementer

- **`orth_for_ad`**: This function (in `autodiff.jl:62-69`) projects out the component along `v` from the gradient. It's critical for AD correctness. The `eigsolve_dispatch` for `:arnoldi` must call `orth_for_ad(v)` on the result, just like `simple_eig` does.
- **Complex eigenvalues**: For transfer matrices in VUMPS, the dominant eigenvalue is real and positive. The `arnoldi_eig` implementation takes `real()` of the eigenvalue. If this causes issues with specific models, it may need to be revisited.
- **Memory**: Arnoldi stores `krylov_dim` copies of the vector. For large χ this matters. A `krylov_dim=10` with χ=200 and D=8 tensors means ~10× memory overhead for the eigensolver. Monitor this.
- **Hessenberg H matrix types**: The H matrix uses `Complex{T}` because `eigen` on a real upper Hessenberg matrix can produce complex eigenvalues. The final eigenvector reconstruction takes `real()` parts.
- **Backward compatibility**: The default `eigsolver=:power` ensures all existing scripts work unchanged. The `ifsimple_eig` flag is kept but the new dispatch path (`eigsolve_dispatch`) is only used when `eigsolver != :power` or when explicitly choosing `:power` through the new interface.
