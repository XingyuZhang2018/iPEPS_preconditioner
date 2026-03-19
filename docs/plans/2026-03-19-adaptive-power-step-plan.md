# Adaptive Power Step Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add adaptive power step scheduling to VUMPS forward phase — power steps double each time err drops an order of magnitude.

**Architecture:** Add `adaptive_power` boolean to VUMPS struct. In `runtime.jl` forward loop, record `err_init` on first iteration, then compute `power_step = clamp(2^floor(log10(err_init/err)), 1, maxiter_power)` each step. AD phase unchanged.

**Tech Stack:** Julia, iPEPS_preconditioner, CUDA (for benchmark)

---

### Task 1: Add `adaptive_power` field to VUMPS struct

**Files:**
- Modify: `src/contraction/interface.jl:22-41`

**Step 1: Add the field**

Add `adaptive_power::Bool = false` to the VUMPS struct, after `maxiter_power_ad`:

```julia
@with_kw mutable struct VUMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    maxiter_ad::Int = Defaults.maxiter_ad
    miniter_ad::Int = Defaults.miniter_ad
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = Defaults.ifsimple_eig
    maxiter_power::Int = 1 # power steps for eigsolve (also k_max when adaptive)
    maxiter_power_ad::Int = 5 # power steps for eigsolve in ad
    adaptive_power::Bool = false # dynamically increase power steps as VUMPS converges
    eigsolver::Symbol = Defaults.eigsolver
    krylov_dim::Int = Defaults.krylov_dim
    krylov_dim_ad::Int = Defaults.krylov_dim
    ifload_env::Bool = true
    ifsave_env::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1
end
```

**Step 2: Commit**

```bash
git add src/contraction/interface.jl
git commit -m "feat: add adaptive_power field to VUMPS struct"
```

---

### Task 2: Implement adaptive scheduling in forward loop

**Files:**
- Modify: `src/contraction/runtime.jl:88-101`

**Step 1: Add adaptive logic to the forward phase loop**

Replace the forward loop in `environment(M, env::VUMPSEnv, alg::VUMPS)` (lines 88-101):

```julia
function environment(M, env::VUMPSEnv, alg::VUMPS)
    t0 = @ignore_derivatives time()
    local err
    @ignore_derivatives begin
        maxiter_power_max = alg.maxiter_power  # save k_max
        err_init = nothing
        for i = 1:alg.maxiter
            env, err = leftmove(M, env, alg)
            # Adaptive power step: double power steps per decade of err reduction
            if alg.adaptive_power
                if err_init === nothing
                    err_init = err
                end
                n_decades = err > 0 && err_init > err ? floor(Int, log10(err_init / err)) : 0
                alg.maxiter_power = clamp(2^n_decades, 1, maxiter_power_max)
            end
            alg.verbosity >= 3 && i % alg.output_interval == 0 && @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\tpow = %d\n", i, time()-t0, err, alg.maxiter_power)
            if err < alg.tol && i >= alg.miniter
                alg.verbosity >= 2 && @info @sprintf("contraction converged@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
                break
            end
            if i == alg.maxiter
                alg.verbosity >= 2 && @warn @sprintf("contraction canceled@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            end
        end
        alg.maxiter_power = maxiter_power_max  # restore k_max
    end

    maxiter_power_origin = alg.maxiter_power
    krylov_dim_origin = alg.krylov_dim
    eigsolver_origin = alg.eigsolver
    alg.maxiter_power = alg.maxiter_power_ad
    alg.krylov_dim = alg.krylov_dim_ad
    # Arnoldi is not Zygote-compatible; use power method for AD phase
    if alg.eigsolver == :arnoldi
        alg.eigsolver = :power
    end
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
    alg.eigsolver = eigsolver_origin

    return env, err
end
```

Key changes from current code:
1. Save `maxiter_power_max` at start, restore after forward loop
2. Record `err_init` on first step
3. Compute `n_decades` and set `alg.maxiter_power` each step
4. Add `pow = %d` to log output so we can see the adaptive schedule
5. Wrap entire forward loop in a single `@ignore_derivatives begin...end` block
6. When `adaptive_power=false`, the `if` block is skipped — zero behavioral change

**Step 2: Commit**

```bash
git add src/contraction/runtime.jl
git commit -m "feat: implement adaptive power step scheduling in VUMPS forward loop"
```

---

### Task 3: Add adaptive configs to benchmark

**Files:**
- Modify: `benchmark/eigsolver_benchmark.jl:39-65`

**Step 1: Extend `make_alg` to support adaptive_power**

Add a new eigsolver type `:adaptive` to `BenchmarkConfig` handling. The `param` field will be `k_max`:

```julia
function make_alg(config::BenchmarkConfig; maxiter, maxiter_ad, miniter_ad, tol, verbosity)
    if config.eigsolver == :power
        return VUMPS(eigsolver=:power,
                     maxiter_power=config.param,
                     maxiter_power_ad=config.param,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false,
                     ifsave_env=false)
    elseif config.eigsolver == :adaptive
        return VUMPS(eigsolver=:power,
                     maxiter_power=config.param,
                     maxiter_power_ad=config.param,
                     adaptive_power=true,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false,
                     ifsave_env=false)
    elseif config.eigsolver == :arnoldi
        # ... unchanged
    end
end
```

**Step 2: Commit**

```bash
git add benchmark/eigsolver_benchmark.jl
git commit -m "feat: add adaptive eigsolver config to benchmark framework"
```

---

### Task 4: Run benchmark — adaptive vs fixed power step

**Step 1: Run GPU benchmark**

```julia
configs = [
    BenchmarkConfig(:power, 5,     "pow5_fixed"),
    BenchmarkConfig(:adaptive, 5,  "adaptive_max5"),
    BenchmarkConfig(:adaptive, 10, "adaptive_max10"),
    BenchmarkConfig(:adaptive, 20, "adaptive_max20"),
]
```

Parameters: D=3, χ=64, 50 optim steps, local preconditioner, seed=42.

**Step 2: Analyze results**

Compare:
- Energy at steps 10, 20, 30, 40, 50
- Total wall time
- Energy vs cumulative time curve

**Step 3: Commit benchmark results summary**

Add results to design doc or a separate results file.

```bash
git add docs/plans/2026-03-19-adaptive-power-step-design.md
git commit -m "docs: add adaptive power step benchmark results"
```
