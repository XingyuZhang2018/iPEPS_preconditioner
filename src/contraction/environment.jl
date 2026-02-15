abstract type AbstractEnv end 
struct CTMEnv{CT <: AbstractArray{<:Number, 2}, ET <: Union{leg3, leg4}} <: AbstractEnv
    C::CT
    T::ET
end

function update!(env::CTMEnv, env´::CTMEnv) 
    env.C .= env´.C
    env.T .= env´.T
    return env
end

struct VUMPSEnv{CT <: AbstractArray{<:Number, 2}, ET <: Union{leg3, leg4}} <: AbstractEnv
    AL::ET
    C::CT
    T::ET
end

function update!(env::VUMPSEnv, env´::VUMPSEnv) 
    env.AL .= env´.AL
    env.C .= env´.C
    env.T .= env´.T
    return env
end

Array(env::VUMPSEnv) = VUMPSEnv(Array(env.AL), Array(env.C), Array(env.T))
CuArray(env::VUMPSEnv) = VUMPSEnv(CuArray(env.AL), CuArray(env.C), CuArray(env.T))
Array(env::CTMEnv) = CTMEnv(Array(env.C), Array(env.T))
CuArray(env::CTMEnv) = CTMEnv(CuArray(env.C), CuArray(env.T))

function CTMenv(Tu, Tl, Td, Tr, M, Cul)
    λ, cul, info = eigsolve(x -> CTMmap(x, Tu, Tl, Td, Tr, M), Cul, 1, :LM)
    info.converged == 0 && error("eigsolve did not converge")
    return λ[1], cul[1]
end

function Cenv(Tu, Td, Cint; alg, ifvalue=false)
    f(C) = Cmap(C, Tu, Td)
    if alg.ifsimple_eig
        λ, v = simple_eig(f, Cint; maxiter=alg.maxiter_power, ifvalue)
        return λ, v
    else
        λs, Cs, info = eigsolve(f, Cint, 1, :LM; maxiter=1)
        info.converged == 0 && error("eigsolve did not converge")
        return λs[1], Cs[1]
    end
end

function Eenv(Tu, Td, M, Tint; alg, ifvalue=false)
    @unpack ifsimple_eig, maxiter_power, forloop_iter, ifparallel, ifcheckpoint = alg
    # f(E) = ifcheckpoint ? checkpoint(FLmap_parallel, E, Tu, Td, M; ifparallel, forloop_iter) : FLmap_parallel(E, Tu, Td, M; ifparallel, forloop_iter) 
    f(E) = FLmap_parallel(E, Tu, Td, M; ifparallel, forloop_iter) 
    if alg.ifsimple_eig
        λ, v = simple_eig(f, Tint; maxiter=maxiter_power, ifvalue)
        return λ, v
    else
        λs, Es, info = eigsolve(f, Tint, 1, :LM; maxiter=1)
        info.converged == 0 && error("eigsolve did not converge")
        return λs[1], Es[1]
    end
end

function ACenv(Tl, Tr, M, Tint; alg, ifvalue=false)
    @unpack ifsimple_eig, maxiter_power, forloop_iter, ifparallel, ifcheckpoint = alg
    # f(AC) = ifcheckpoint ? checkpoint(ACmap_parallel, AC, Tl, Tr, M; ifparallel, forloop_iter) :  ACmap_parallel(AC, Tl, Tr, M; ifparallel, forloop_iter)
    f(AC) = ACmap_parallel(AC, Tl, Tr, M; ifparallel, forloop_iter)
    if alg.ifsimple_eig
        λ, v = simple_eig(f, Tint; maxiter=maxiter_power, ifvalue)
        return λ, v
    else
        λs, Es, info = eigsolve(f, Tint, 1, :LM; maxiter=1)
        info.converged == 0 && error("eigsolve did not converge")
        return λs[1], Es[1]
    end
end

function getU(env::CTMEnv, ::QRCTM)
    @unpack C, T = env
    Tu = CTtoT(C, T)
    U, C = qr_for_ad(_to_front(Tu))
    U = reshape(U, size(T))
    return U, C
end

function leftmove(M, env::CTMEnv, alg::QRCTM)
    @unpack C, T = env

    CT = _to_front(CTtoT(C, T))
    U, R = qr_for_ad(CT)
    U = reshape(U, size(T))

    T = FLmap_parallel(T, U, U, M; ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter)
    C′ = Cmap(R, T, U)

    T /= Zygote.@ignore norm(T)
    C′ /= Zygote.@ignore norm(C′)
    err = Zygote.@ignore norm(C′ - C)
    
    return CTMEnv(C′, T), err
end

function leftmove(M, env::VUMPSEnv, alg::VUMPS)
    @unpack AL, C, T = env

    AC = ALCtoAC(AL, C)
    _, T = Eenv(AL, conj(AL), M, T; alg)
    _, AC = ACenv(T, T, M, AC; alg)
    _, C = Cenv(T, T, C; alg)

    QAC, RAC = qrpos(_to_front(AC))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', size(AC))
    err = Zygote.@ignore norm(RAC - RC)

    return VUMPSEnv(AL, C, T), err
end