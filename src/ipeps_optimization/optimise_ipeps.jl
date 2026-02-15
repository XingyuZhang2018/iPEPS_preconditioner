
"""
optimise_ipeps(A, χ, χshift::Int, model, params::iPEPSOptimize; restriction_ipeps=_restriction_ipeps)

Run L-BFGS optimization of an iPEPS starting from tensor set `A`.

Behavior:
- Initializes the boundary environment at bond dimension `χ` and runs the
    chosen optimizer (via `optimize_reload`) to minimize the variational energy
    of `A` with respect to `model`.
- Supports optional preconditioning strategies controlled by `params.preconditiontype`.
- After each optimization stage the boundary dimension `χ` is incremented by
    `χshift` and environments, checkpoints and observables are saved under
    `params.folder` according to the I/O settings in `params`.
- The optional `restriction_ipeps` function is used to map parameters into a
    restricted iPEPS representation when required.

Arguments:
- `A`: initial iPEPS tensors or parameter vector for the optimizer.
- `χ`: initial boundary/environment bond dimension.
- `χshift`: integer increment applied to `χ` after each stage.
- `model`: Hamiltonian or model used to evaluate the energy/observables.
- `params`: an `iPEPSOptimize` struct controlling optimizer, preconditioning,
    verbosity and I/O behavior.

Side effects:
- Saves environment files, L-BFGS checkpoints and iPEPS snapshots to disk.
- Prints progress and history logs according to `params.verbosity`.

Return:
- This function completes when optimization finishes; it does not explicitly
    guarantee a return value. Optimized tensors and artifacts are written to disk.
"""
function optimise_ipeps(A, χ, χshift::Int, model, params::iPEPSOptimize;
                                                restriction_ipeps = _restriction_ipeps)
    D = size(A, 1)

    A′ = restriction_ipeps(A)
    env = initialize_env(A′, χ, params.boundary_alg; file=joinpath(params.folder, "D$D", "env", "$χ.jld2"))
    env′ = deepcopy(env)
    fδEierr = [1.0,1.0,0.0,0.0]

    function f(A)
        A = restriction_ipeps(A)
        return real(energy(A, model, env, env′, fδEierr, params))
    end
    function fg(x)
        t1 = time()
        e, vjp = pullback(f, x)
        params.verbosity >= 2 && printstyled(" forward calculation took $(round(time() - t1, digits = 2)) s\n"; bold=true, color=:green) 
        reclaim(x)
        t2 = time()
        g = vjp(1)[1]
        params.verbosity >= 2 && printstyled("backward calculation took $(round(time() - t2, digits = 2)) s\n"; bold=true, color=:green)
        reclaim(g)
        return e, g
    end
    @unpack optimizer, iter_precond, preconditiontype = params
    t0 = time()

    function precondition(x, g)
        if preconditiontype == :none
            g
        elseif preconditiontype == :local
            precondition_inverse_local_metric(x, g, env′, params, restriction_ipeps, fδEierr, params.iter_precond)
        elseif preconditiontype == :full
            precondition_inverse_full_metric(x, g, env′, params, restriction_ipeps, fδEierr, params.iter_precond)
        elseif preconditiontype == :BP
            precondition_invese_BP(A, grad, restriction_ipeps, fδEi, iter_precond)
        else
            error("Unknown preconditioner type: $preconditiontype")
        end
    end

    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")

    for _ in 1:100
        resume_from = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing
        save_state_to = params.ifsave_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing
        A, e, eg, numfg, history = optimize_reload(fg, A, optimizer; 
                                                   resume_from,
                                                   save_state_to,
                                                   save_interval=params.save_interval,
                                                   precondition, 
                                                   inner = _inner, 
                                                   finalize! = (x, f, g, iter)->_finalize!(x, f, g, iter, env, env′, D, χ, params, t0, fδEierr)
        )
        χ += χshift 
        params_obs = deepcopy(params)
        params_obs.boundary_alg.maxiter = params.boundary_alg.maxiter * 10
        enew,  = observable(A, χ, model, params_obs; restriction_ipeps)
        env = initialize_env(A′, χ, params_obs.boundary_alg; file=joinpath(params.folder, "D$D", "env", "$χ.jld2"))
        env′ = deepcopy(env)
        if abs(real(enew[1]) - e) < 1e-7 && history[end-1] < 1e-5
            break
        end
    end
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))
function _finalize!(x, f, g, iter, env, env′, D, χ, params, t0, fδEierr)
    @unpack folder = params

    fδEierr[3] = iter
    fδEierr[2] = abs(fδEierr[1] - f)
    fδEierr[1] = f
    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy_χ%d = %.15f \tgnorm = %.3e\tEimag = %.3e\n", iter, time() - t0, χ, f, norm(g), fδEierr[4])

    folder1 = joinpath(folder, "D$(D)")
    !(ispath(folder1)) && mkpath(folder1)
    params.reuse_env && update!(env, env′)
    if params.boundary_alg.ifsave_env
        file = joinpath(params.folder, "D$D", "env", "χ$χ.jld2")
        save_env(file, env)
        params.verbosity >= 2 && @info "Saved environment to $file"
    end

    if params.verbosity >= 3 && iter % params.output_interval == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder1, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_interval != 0 && iter % params.save_interval == 0
        save(joinpath(folder1, "ipeps", "χ$(χ)", "No.$(iter).jld2"), "bcipeps", Array(x))
    end
    
    if fδEierr[2] < 1e-12 || fδEierr[4] > 1e-7
        g .= 0
    end
    return x, f, g
end 