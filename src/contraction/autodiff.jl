@non_differentiable initialize_env(kwargs...)

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NoTangent(), dA
    end
    return As, back
end

function ChainRulesCore.rrule(::typeof(qr_for_ad), A::AbstractArray{T,2}) where {T}
    Q, R = qr_for_ad(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        dA = (dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * 1e-12)'
        return NoTangent(), _arraytype(A)(dA)
    end
    return (Q, R), back
end

function qr_for_ad(A::AbstractMatrix{<:ForwardDiff.Dual})
    A_val = ForwardDiff.value.(A)
    dA = ForwardDiff.partials.(A, 1)

    Q, R = qr_for_ad(A_val)
    K = (Q' * dA) / UpperTriangular(R + I * 1e-12)

    U = triu(K)
    dR = U * R

    H = K - U  
    Ω = H - H' 
    
    dQ = Q * Ω

    T = ForwardDiff.tagtype(eltype(A))
    Q_dual = ForwardDiff.Dual{T}.(Q, dQ)
    R_dual = ForwardDiff.Dual{T}.(R, dR)

    return Q_dual, R_dual
end
    
function ChainRulesCore.rrule(::Type{<:CTMEnv}, C, T)
    env = CTMEnv(C, T)
    function back(∂env)
        ∂C, ∂T = ∂env
        return NoTangent(), ∂C, ∂T 
    end
    return env, back
end

function ChainRulesCore.rrule(::Type{<:VUMPSEnv}, AL, C, T)
    env = VUMPSEnv(AL, C, T)
    function back(∂env)
        ∂AL, ∂C, ∂T = ∂env
        return NoTangent(), ∂AL, ∂C, ∂T 
    end
    return env, back
end

orth_for_ad(v) = v
function ChainRulesCore.rrule(::typeof(orth_for_ad), v)
    function back(dv)
        dv .-= dot(v, dv) * v
        return NoTangent(), dv
    end
    return v, back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        dA = (dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * 1e-12)'
        return NoTangent(), _arraytype(A)(dA)
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(forloop), f, args...; forloop_iter, N_in, N_out, size_out)
    if forloop_iter == 1
        result, back = pullback(f, args...)
        function realback(dresult)
            dargs = back(dresult)
            return NoTangent(), NoTangent(), dargs...
        end
        return result, realback
    else
        Ain = args[N_in[1]]
        split_dim = N_in[2]
        D_split = size(Ain, split_dim)

        result = similar(args[1], size_out)

        in_idx  = ntuple(_ -> (:), ndims(Ain))
        out_idx = ntuple(_ -> (:), ndims(result))

        ranges = split_ranges(D_split, forloop_iter)

        @views for r in ranges
            in_idx_r  = Base.setindex(in_idx,  r, split_dim)
            out_idx_r = Base.setindex(out_idx, r, N_out)
            split_args = ntuple(length(args)) do j
                j == N_in[1] ? view(args[j], in_idx_r...) : args[j]
            end

            result[out_idx_r...] .= f(split_args...)
        end

        function back(dresult)
            dargs = ntuple(i->args[i] isa Tuple ? zero.(args[i]) : zero(args[i]), length(args))
            @views for r in ranges
                in_idx_r  = Base.setindex(in_idx,  r, split_dim)
                out_idx_r = Base.setindex(out_idx, r, N_out)
                split_args = ntuple(length(args)) do j
                    j == N_in[1] ? view(args[j], in_idx_r...) : args[j]
                end    
                _, bp = pullback(f, split_args...)
                dargs_range = bp(view(dresult, out_idx_r...))
                for i in 1:length(args)
                    if i == N_in[1]
                        dargs[i][in_idx_r...] .= dargs_range[i]
                    else
                        if dargs_range[i] isa Tuple
                            for j in 1:length(dargs_range[i])
                                dargs[i][j] .+= dargs_range[i][j]
                            end
                        else
                            dargs[i] .+= dargs_range[i]
                        end
                    end
                end
            end
            return NoTangent(), NoTangent(), dargs...
        end

        return result, back
    end
end

function ChainRulesCore.rrule(::typeof(parallel), f, args...; forloop_iter, N_in, N_out, size_out)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    D_split = size(args[N_in[1]])[N_in[2]]
    result = similar(args[1], size_out)
    D_split_ranges = split_ranges(D_split, nprocs*forloop_iter)

    for i in 1:forloop_iter
        ind = forloop_iter * rank + i
        cols_in = (j == N_in[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in[1]]))
        cols_out = (j == N_out ? D_split_ranges[ind] : (:) for j in 1:ndims(result))
        split_args = Tuple(j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args))
        result[cols_out...] = f(split_args...)
        synchronize(args[1])
    end

    element_size = prod(size_out) ÷ D_split
    counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
    MPI.Allgatherv!(VBuffer(result, counts), comm)

    function back(dresult)
        dargs = ntuple(i->args[i] isa Tuple ? zero.(args[i]) : zero(args[i]), length(args))
        for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            cols_in = (j == N_in[2] ? D_split_ranges[ind] : (:) for j in 1:ndims(args[N_in[1]]))
            cols_out = (j == N_out ? D_split_ranges[ind] : (:) for j in 1: ndims(result))
            split_args = Tuple(j == N_in[1] ? args[j][cols_in...] : args[j] for j in 1:length(args))
            _, bp = pullback(f, split_args...)
            split_dargs = bp(dresult[cols_out...])
            for j in 1:length(args)
                if j == N_in[1] 
                    dargs[j][cols_in...] = split_dargs[j]
                else
                    if dargs[j] isa Tuple
                        for k in 1:length(dargs[j])
                            dargs[j][k] .+= split_dargs[j][k]
                        end
                    else
                        dargs[j] .+= split_dargs[j]
                    end
                end
            end
        end

        synchronize(args[1])

        for j in 1:length(args)
            if j == N_in[1] 
                element_size = prod(size(dargs[j])) ÷ D_split
                counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
                MPI.Allgatherv!(VBuffer(dargs[j], counts), comm)
            else
                if dargs[j] isa Tuple
                    for k in 1:length(dargs[j])
                        MPI.Allreduce!(dargs[j][k], +, comm)
                    end
                else
                    MPI.Allreduce!(dargs[j], +, comm)
                end
            end
        end
        
        return NoTangent(), NoTangent(), dargs...
    end

    return result, back
end

function fix_gauge_leftmove(M, env::CTMEnv, alg::QRCTM)
    @unpack C, T = env
    for _ in 1:2
        CT = _to_front(CTtoT(C, T))
        U, R = qr_for_ad(CT)
        U = reshape(U, size(T))

        T = FLmap_parallel(T, U, U, M; ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter)
        C = Cmap(R, T, U)
        T /= Zygote.@ignore norm(T)
        C /= Zygote.@ignore norm(C)
    end

    return CTMEnv(C, T)
end

function ChainRulesCore.rrule(::typeof(environment_fix_point),
                              M,
                              env::CTMEnv,
                              alg::Algorithm)

    #forward
    env_out, err = environment(M, env, alg)

    function back((∂env_out, ∂err))
        # -----------------------------
        # checkpoint: only ONE pullback
        # -----------------------------
        _, pb = pullback(fix_gauge_leftmove, M, env_out, alg)
        # @show norm(env_new.T - env_out.T)
        # @show norm(env_new.C - env_out.C)

        # VJP helpers
        vjp_env_env(∂env) = Tangent{Any}(; pb(∂env)[2]...)
        vjp_env_M(∂env)   = pb(∂env)[1]

        # -----------------------------
        # Neumann series (functional!)
        # ∂envsum = Σ_k (J_env^*)^k ∂env
        # -----------------------------
        ∂env_k   = ∂env_out
        ∂envsum  = ZeroTangent()
        ϵ = norm(∂env_k)
        ∂envsum += ∂env_k

        for ix in 1:10000
            ∂env_k   = vjp_env_env(∂env_k)
            ϵ′ = norm(∂env_k)
            if ϵ′ < 1e-12 || ϵ′ > ϵ
                break
            end
            ϵ = ϵ′
            ∂envsum += ∂env_k
        end

        # -----------------------------
        # propagate back to M
        # -----------------------------
        ∂M = vjp_env_M(∂envsum)

        return NoTangent(), ∂M, NoTangent(), NoTangent()
    end

    return (env_out, err), back
end

project_AL(∂AL, AL) = project_AL!(deepcopy(∂AL), AL)
function project_AL!(∂AL, AL)
    # ∂AL .-= ein"deg,(abc,abg)->dec"(AL, ∂AL, conj.(AL))
    @tensoropt out[d,e,h,c] := AL[a,b,f,c] * ∂AL[a,b,f,g] * conj(AL[d,e,h,g])
    ∂AL .-= out
    return ∂AL
end

function fix_gauge_leftmove(M, env::VUMPSEnv, alg::VUMPS)
    @unpack AL, C, T = env

    AC = ALCtoAC(AL, C)
    _, T = Eenv(AL, conj(AL), M, T; alg)
    _, AC = ACenv(T, T, M, AC; alg)
    _, C = Cenv(T, T, C; alg)

    QAC, _ = qrpos(_to_front(AC))
    QC,_ = qrpos(C)
    AL = reshape(QAC*QC', size(AC))

    T /= Zygote.@ignore norm(T)
    C /= Zygote.@ignore norm(C)

    return VUMPSEnv(AL, C, T)
end

function ChainRulesCore.rrule(::typeof(environment_fix_point),
                              M,
                              env::VUMPSEnv,
                              alg::VUMPS)

    #forward
    env_out, err = environment(M, env, alg)

    function back((∂env_out, ∂err))
        # ∂AL, ∂C, ∂T = ∂env_out.AL, ∂env_out.C, ∂env_out.T
        # ∂AL = project_AL(∂AL, env_out.AL)
        # ∂env_out = Tangent{Any}(; AL=∂AL, C=∂C, T=∂T)
        # -----------------------------
        # checkpoint: only ONE pullback
        # -----------------------------
        _, pb = pullback(fix_gauge_leftmove, M, env_out, alg)
        # @show norm(env_new.T - env_out.T)
        # @show norm(env_new.C - env_out.C)
        # @show norm(env_new.AL - env_out.AL)

        # VJP helpers
        # function vjp_env_env(∂env)
        #     ∂AL, ∂C, ∂T = ∂env.AL, ∂env.C, ∂env.T
        #     ∂AL = project_AL(∂AL, env_out.AL)
        #     ∂env = Tangent{Any}(; AL=∂AL, C=∂C, T=∂T)
        #     ∂env = pb(∂env)[2]
        #     ∂AL, ∂C, ∂T = ∂env.AL, ∂env.C, ∂env.T
        #     ∂AL = project_AL(∂AL, env_out.AL)
        #     return Tangent{Any}(; AL=∂AL, C=∂C, T=∂T)
        # end
        vjp_env_env(∂env) = Tangent{Any}(; pb(∂env)[2]...)
        vjp_env_M(∂env)   = pb(∂env)[1]

        # -----------------------------
        # Neumann series (functional!)
        # ∂envsum = Σ_k (J_env^*)^k ∂env
        # -----------------------------
        ∂env_k   = ∂env_out
        ∂envsum  = ZeroTangent()
        ϵ = norm(∂env_k)
        ∂envsum += ∂env_k

        for ix in 1:100
            ∂env_k = vjp_env_env(∂env_k)
            ϵ′ = norm(∂env_k)
            if ϵ′ < 1e-12 || ϵ′ > ϵ
                break
            end
            ϵ = ϵ′
            ∂envsum += ∂env_k
        end

        # -----------------------------
        # propagate back to M
        # -----------------------------
        ∂M = vjp_env_M(∂envsum)

        return NoTangent(), ∂M, NoTangent(), NoTangent()
    end

    return (env_out, err), back
end
