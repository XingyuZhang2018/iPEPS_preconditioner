function observable(A, χ, model, params::iPEPSOptimize;
                    restriction_ipeps = _restriction_ipeps)
    D = size(A, 1)

    A = restriction_ipeps(A)
    env = initialize_env(A, χ, params.boundary_alg; file=joinpath(params.folder, "D$D", "env", "$χ.jld2"))
    env, _ = environment(A, env, params.boundary_alg)
    if params.boundary_alg.ifsave_env
        file = joinpath(params.folder, "D$D", "env", "$χ.jld2")
        save_env(file, env)
        params.verbosity >= 2 && @info "Saved environment to $file"
    end

    e = expectation_value(A, model, env, params)
    mag = magnetization_value(A, env, params)
    ξ = cor_len_value(env, params)
    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$D"), params)
    return e, mag, ξ
end

function magnetization_value(A, env::VUMPSEnv, params)
    @unpack AL, C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    m_dict = Dict{String, Any}()
    etype = eltype(A)
    atype = _arraytype(A)

    AC = ALCtoAC(AL,C)
    Mx = contract_o1(T, AC, A, atype(Sx); ifparallel, forloop_iter)
    My = etype == Float64 ? 0.0 : contract_o1(T, AC, A, atype(Sy); ifparallel, forloop_iter)
    Mz = contract_o1(T, AC, A, atype(Sz); ifparallel, forloop_iter)

    n = contract_n1(T, AC, A; ifparallel, forloop_iter)
    Mag = [Mx/n, My/n, Mz/n]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")
    m_dict["1,1"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm)

    return Mnorm, m_dict
end

function magnetization_value(A, env::CTMEnv, params)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    m_dict = Dict{String, Any}()
    etype = eltype(A)
    atype = _arraytype(A)

    To = CTCtoT(C, T)
    Mx = contract_o1(To, T, A, atype(Sx); ifparallel, forloop_iter)
    My = etype == Float64 ? 0.0 : contract_o1(To, T, A, atype(Sy); ifparallel, forloop_iter)
    Mz = contract_o1(To, T, A, atype(Sz); ifparallel, forloop_iter)

    n = contract_n1(To, T, A; ifparallel, forloop_iter)
    Mag = [Mx/n, My/n, Mz/n]
    Mnorm = norm(Mag)
    params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")
    m_dict["1,1"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm)

    return Mnorm, m_dict
end

function cor_len_value(env::CTMEnv, params) 
    @unpack C, T = env

    λcs, _, info = eigsolve(C->Lmap(C, T, conj(T)), C, 10, :LM; maxiter=100, ishermitian = false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
        
    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")

    return ξ
end

function cor_len_value(env::VUMPSEnv, params) 
    @unpack AL, C, T = env

    λcs, _, info = eigsolve(C->Lmap(C, AL, conj(AL)), C, 10, :LM; maxiter=100, ishermitian = false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
        
    ξ = -1/log(abs(λ2/λcs[1]))
    @show ξ
    params.verbosity >= 4 && println("ξ = $(ξ)")

    return ξ
end

function write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)
    path = joinpath(folder, "observable")
    isdir(path) || mkpath(path)
    obs_log = joinpath(path, "χ$χ.log")
    e_dict = e[2]
    m_dict = mag[2]

    open(obs_log, "w") do io
        write(io, @sprintf("energy_per_site:\n%.15f\n", real(e[1])))

        for bond_type in keys(e_dict)
            write(io, "$bond_type: i j energy\n")
            for pos in keys(e_dict[bond_type])
                write(io, @sprintf("%s %.15f\t", pos, real(e_dict[bond_type][pos])))
            end
            write(io, "\n")
        end
        
        write(io, @sprintf("magnetization_norm_per_site:\n%.15f\n", real(mag[1])))

        write(io, "magnetization: i j |M| Mx My Mz\n")
        for pos in keys(m_dict)
            write(io, @sprintf("%s %.15f %.15f %.15f %.15f\t", pos, real(m_dict[pos]["|M|"]), real(m_dict[pos]["Mx"]), real(m_dict[pos]["My"]), real(m_dict[pos]["Mz"])))
            write(io, "\n")
        end
        write(io, @sprintf("correlation_length:\n%.15f\n", real(ξ)))
    end
end