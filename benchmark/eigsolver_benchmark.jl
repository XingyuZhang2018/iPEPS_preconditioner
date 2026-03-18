using iPEPS_preconditioner
using Random
using LinearAlgebra
using JLD2
using Printf
using Dates
using OptimKit

struct BenchmarkConfig
    eigsolver::Symbol
    param::Int          # maxiter_power for :power, krylov_dim for :arnoldi
    label::String
end

struct StepRecord
    iter::Int
    time::Float64
    energy::Float64
    gnorm::Float64
end

function parse_history_log(logfile::String)
    records = StepRecord[]
    isfile(logfile) || return records
    for line in eachline(logfile)
        m = match(r"i\s*=\s*(\d+)\s*t\s*=\s*([\d.]+)\s*sec\s*energy_χ\d+\s*=\s*([-\d.]+)\s*gnorm\s*=\s*([\d.eE+-]+)", line)
        if m !== nothing
            push!(records, StepRecord(
                parse(Int, m[1]),
                parse(Float64, m[2]),
                parse(Float64, m[3]),
                parse(Float64, m[4])
            ))
        end
    end
    return records
end

function make_alg(config::BenchmarkConfig; maxiter, maxiter_ad, miniter_ad, tol, verbosity)
    if config.eigsolver == :power
        return VUMPS(eigsolver=:power,
                     maxiter_power=config.param,
                     maxiter_power_ad=config.param,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false,
                     ifsave_env=false)
    elseif config.eigsolver == :arnoldi
        return VUMPS(eigsolver=:arnoldi,
                     krylov_dim=config.param,
                     krylov_dim_ad=config.param,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false,
                     ifsave_env=false)
    elseif config.eigsolver == :krylovkit
        return VUMPS(eigsolver=:krylovkit,
                     maxiter=maxiter, maxiter_ad=maxiter_ad,
                     miniter_ad=miniter_ad, tol=tol,
                     verbosity=verbosity, ifload_env=false,
                     ifsave_env=false, ifsimple_eig=false)
    else
        error("Unknown eigsolver: $(config.eigsolver)")
    end
end

function run_benchmark(config::BenchmarkConfig;
                       seed=42, D=2, χ=10, χshift=0,
                       maxiter_boundary=10, maxiter_ad=4, miniter_ad=4,
                       tol_boundary=1e-10,
                       n_optim_steps=30,
                       atype=Array)
    Random.seed!(seed)
    model = Heisenberg()

    boundary_alg = make_alg(config;
                            maxiter=maxiter_boundary, maxiter_ad=maxiter_ad,
                            miniter_ad=miniter_ad, tol=tol_boundary,
                            verbosity=3)

    folder = mktempdir()
    params = GradientOptimize(
        boundary_alg=boundary_alg,
        iffixedpoint=false,
        optimizer=LBFGS(10; maxiter=n_optim_steps, verbosity=3, gradtol=1e-12),
        reuse_env=true,
        verbosity=3,
        folder=folder,
        preconditiontype=:none,
        ifload_lbfgs=false,
        iter_precond=0,
        output_interval=1,
        save_interval=9999,
        ifsave_lbfgs=false,
    )

    A = init_ipeps(; atype, No=0, d=2, D, χ, params)

    t_start = time()
    optimise_ipeps(A, χ, χshift, model, params)
    total_time = time() - t_start

    logfile = joinpath(folder, "D$D", "history.log")
    records = parse_history_log(logfile)
    final_energy = isempty(records) ? NaN : records[end].energy

    @printf("  %-20s  steps=%3d  total=%.1fs  final_E=%.10f\n",
            config.label, length(records), total_time, final_energy)

    return (config=config, records=records, total_time=total_time, final_energy=final_energy)
end

function print_comparison_table(results)
    println("\n" * "=" ^ 80)
    println("Summary")
    println("=" ^ 80)
    @printf("%-20s  %6s  %10s  %18s  %12s\n", "Config", "Steps", "Total(s)", "Final Energy", "Avg dt(s)")
    @printf("%-20s  %6s  %10s  %18s  %12s\n", "-"^20, "-"^6, "-"^10, "-"^18, "-"^12)

    for r in results
        avg_dt = isempty(r.records) ? NaN : r.total_time / length(r.records)
        @printf("%-20s  %6d  %10.2f  %18.12f  %12.3f\n",
                r.config.label, length(r.records), r.total_time, r.final_energy, avg_dt)
    end
end

function print_energy_trajectory(results; max_steps=15)
    println("\n" * "=" ^ 80)
    println("Energy trajectory (first $max_steps steps)")
    println("=" ^ 80)

    header = @sprintf("%-5s", "Step")
    for r in results
        header *= @sprintf("  %18s", r.config.label[1:min(18,end)])
    end
    println(header)
    println("-"^(5 + 20*length(results)))

    for step in 1:max_steps
        line = @sprintf("%-5d", step)
        for r in results
            if step <= length(r.records)
                line *= @sprintf("  %18.12f", r.records[step].energy)
            else
                line *= @sprintf("  %18s", "—")
            end
        end
        println(line)
    end
end

function print_time_per_step(results; max_steps=10)
    println("\n" * "=" ^ 80)
    println("Wall time per step (seconds)")
    println("=" ^ 80)

    header = @sprintf("%-5s", "Step")
    for r in results
        header *= @sprintf("  %12s", r.config.label[1:min(12,end)])
    end
    println(header)
    println("-"^(5 + 14*length(results)))

    for step in 1:max_steps
        line = @sprintf("%-5d", step)
        for r in results
            if step <= length(r.records)
                dt = step == 1 ? r.records[1].time : r.records[step].time - r.records[step-1].time
                line *= @sprintf("  %12.2f", dt)
            else
                line *= @sprintf("  %12s", "—")
            end
        end
        println(line)
    end
end

function main(; D=2, χ=10, n_optim_steps=30, seed=42,
                maxiter_boundary=10, maxiter_ad=4, miniter_ad=4)
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

    println("=" ^ 80)
    println("Eigsolver Benchmark — $(now())")
    println("D=$D, χ=$χ, n_optim_steps=$n_optim_steps, seed=$seed")
    println("maxiter_boundary=$maxiter_boundary, maxiter_ad=$maxiter_ad, miniter_ad=$miniter_ad")
    println("=" ^ 80)

    results = []
    for (i, config) in enumerate(configs)
        @printf("\n[%2d/%2d] Running: %s\n", i, length(configs), config.label)
        result = run_benchmark(config; seed, D, χ, n_optim_steps,
                               maxiter_boundary, maxiter_ad, miniter_ad)
        push!(results, result)
    end

    print_comparison_table(results)
    print_energy_trajectory(results)
    print_time_per_step(results)

    outfile = "benchmark/results_D$(D)_chi$(χ)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2"
    mkpath(dirname(outfile))
    @save outfile results
    println("\nResults saved to $outfile")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
