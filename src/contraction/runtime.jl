function initialize_env(M, χ, alg::Algorithm; file::String="env.jld2")
    if alg.ifload_env && ispath(file)
        env = load_env(file, _arraytype(M))
        alg.verbosity >= 2 && @info "load CTM env from $file"
        return env
    else
        D = size(M,1)
        if M isa leg4
            T = rand!(similar(M,χ,D,χ))
            T += conj(permutedims(T, (3,2,1)))
        else
            T = rand!(similar(M,χ,D,D,χ))
            T += conj(permutedims(T, (4,2,3,1)))
        end
        C = rand!(similar(M,χ,χ))
        C += C'
        _, C = Cenv(T, conj(T), C; alg, ifvalue=false)
        if alg.verbosity >= 4
            printstyled("start $alg random initial environment->  \n"; bold=true, color=:green) 
        elseif alg.verbosity >= 2
            printstyled("start random initial χ$(χ) CTM environment->  \n"; bold=true, color=:green) 
        end
        return CTMEnv(C, T)
    end
end

function initialize_env(M, χ, alg::VUMPS; file::String="env.jld2")
    if alg.ifload_env && ispath(file)
        env = load_env(file, _arraytype(M))
        alg.verbosity >= 2 && @info "load VUMPS env from $file"
        return env
    else
        D = size(M,1)
        if M isa leg4
            T = rand!(similar(M,χ,D,χ))
            T += conj(permutedims(T, (3,2,1)))
        else
            T = rand!(similar(M,χ,D,D,χ))
            T += conj(permutedims(T, (4,2,3,1)))
        end
        AL, C = qr(_to_front(T))
        AL = reshape(_arraytype(T)(AL), size(T))
        # C += C'
        # _, C = Cenv(T, conj(T), C; alg, ifvalue=false)
        if alg.verbosity >= 4
            printstyled("start $alg random initial environment->  \n"; bold=true, color=:green) 
        elseif alg.verbosity >= 2
            printstyled("start random initial χ$(χ) VUMPS environment->  \n"; bold=true, color=:green) 
        end
        return VUMPSEnv(AL, C, T)
    end
end

environment_fix_point(M, env, alg) = environment(M, env, alg)

function environment(M, env::CTMEnv, alg::Algorithm)
    # freenergy = logZ(M, env)
    t0 = @ignore_derivatives time()
    err = Inf
    local λ
    @ignore_derivatives for i = 1:alg.maxiter
        env, err = leftmove(M, env, alg)
        alg.verbosity >= 3 && i % alg.output_interval == 0 && @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && @info @sprintf("contraction converged@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && @warn @sprintf("contraction canceled@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        end
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

    return env, err
end

function environment(M, env::VUMPSEnv, alg::VUMPS)
    t0 = @ignore_derivatives time()
    local err
    @ignore_derivatives for i = 1:alg.maxiter
        env, err = leftmove(M, env, alg)
        alg.verbosity >= 3 && i % alg.output_interval == 0 && @info @sprintf("i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && @info @sprintf("contraction converged@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && @warn @sprintf("contraction canceled@i = %5d,\tt = %.2fs\terr = %.3e\n", i, time()-t0, err)
        end
    end

    maxiter_power_origin = alg.maxiter_power
    alg.maxiter_power = alg.maxiter_power_ad
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

    return env, err
end