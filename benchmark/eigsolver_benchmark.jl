using iPEPS_preconditioner
using Random
using LinearAlgebra
using JLD2
using Printf
using Dates

struct BenchmarkConfig
    eigsolver::Symbol
    param::Int
    label::String
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
    optimise_ipeps(A, χ, χshift, model, params)
    total_time = time() - t_start

    @printf("  %20s  total_time=%.2fs\n", config.label, total_time)

    return (config=config, total_time=total_time)
end

function main()
    configs = BenchmarkConfig[
        BenchmarkConfig(:power, 1,  "power_k=1"),
        BenchmarkConfig(:power, 2,  "power_k=2"),
        BenchmarkConfig(:power, 5,  "power_k=5"),
        BenchmarkConfig(:power, 10, "power_k=10"),
        BenchmarkConfig(:power, 20, "power_k=20"),
        BenchmarkConfig(:arnoldi, 3,  "arnoldi_k=3"),
        BenchmarkConfig(:arnoldi, 5,  "arnoldi_k=5"),
        BenchmarkConfig(:arnoldi, 10, "arnoldi_k=10"),
        BenchmarkConfig(:arnoldi, 15, "arnoldi_k=15"),
        BenchmarkConfig(:arnoldi, 20, "arnoldi_k=20"),
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

    outfile = "benchmark/results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2"
    mkpath(dirname(outfile))
    @save outfile results
    println("\nResults saved to $outfile")

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
