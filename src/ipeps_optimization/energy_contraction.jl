function energy(A, model, env, env′, fδEierr, params::iPEPSOptimize)
    if params.iffixedpoint 
        env, _ = environment_fix_point(A, env, params.boundary_alg)
    else
        env, _ = environment(A, env, params.boundary_alg)
    end
    Zygote.@ignore begin
        update!(env′, env)
        
        if env isa CTMEnv
            @unpack C, T = env
            iSy = _arraytype(T)(real(1im*Sy))
            To = CTCtoT(C, T)
            fδEierr[4] = abs(contract_o1(To, T, A, iSy; forloop_iter = params.boundary_alg.forloop_iter, ifparallel = params.boundary_alg.ifparallel))
        elseif env isa VUMPSEnv
            @unpack AL, C, T = env
            iSy = _arraytype(T)(real(1im*Sy))
            AC = ALCtoAC(AL, C)
            fδEierr[4] = abs(contract_o1(T, AC, A, iSy; forloop_iter = params.boundary_alg.forloop_iter, ifparallel = params.boundary_alg.ifparallel))
        end
    end
    return expectation_value(A, model, env, params)[1]
end

function expectation_value(A, model::Heisenberg, env::CTMEnv, params::iPEPSOptimize)
    @unpack C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
    )

    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    D,d = size(A)[[1,5]]
    Dh = size(h1, 3)

    # A1u = reshape(ein"abcde,efi->abcidf"(A, h1), D,D,D*Dh,D,d)
    # A2u = reshape(ein"abcde,ief->aibcdf"(A, h2), D*Dh,D,D,D,d)
    A1u = reshape((@tensor A1u[a,b,c,i,d,f] := A[a,b,c,d,e] * h1[e,f,i]), D,D,D*Dh,D,d)
    A2u = reshape((@tensor A2u[a,i,b,c,d,f] := A[a,b,c,d,e] * h2[i,e,f]), D*Dh,D,D,D,d)

    To = CTCtoT(C, T)
    e = oc_H_leg4(To, T, T, A1u, A, A2u, A; ifparallel, forloop_iter)
    n = oc_H_leg4(To, T, T, A, A, A, A; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end

function expectation_value(A, model::Heisenberg, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack AL, C, T = env
    @unpack ifparallel, forloop_iter = params.boundary_alg
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
    )

    h1, h2 = Zygote.@ignore _arraytype(A).(hamiltonian_trunc(model))
    D,d = size(A)[[1,5]]
    Dh = size(h1, 3)

    # A1u = reshape(ein"abcde,efi->abcidf"(A, h1), D,D,D*Dh,D,d)
    # A2u = reshape(ein"abcde,ief->aibcdf"(A, h2), D*Dh,D,D,D,d)
    A1u = reshape((@tensor A1u[a,b,c,i,d,f] := A[a,b,c,d,e] * h1[e,f,i]), D,D,D*Dh,D,d)
    A2u = reshape((@tensor A2u[a,i,b,c,d,f] := A[a,b,c,d,e] * h2[i,e,f]), D*Dh,D,D,D,d)
    
    AC = ALCtoAC(AL,C)
    e = oc_H_leg4(T, AC, AL, A1u, A, A2u, A; ifparallel, forloop_iter)
    n = oc_H_leg4(T, AC, AL, A, A, A, A; ifparallel, forloop_iter)
    params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
    etol = e/n
    e_dict["Horizontal_energy"]["1,1"] = e/n

    params.verbosity >= 3 && println("energy = $(etol*2)")
    return etol*2, e_dict
end
