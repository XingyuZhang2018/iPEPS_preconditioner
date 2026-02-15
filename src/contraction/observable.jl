function logZ(M::leg4, env::CTMEnv)
    @unpack C, T = env

    # alternative implementation easier but slow 
    # λM, _ = Eenv(T, conj(T), M, T; ifsimple_eig=false)
    # λN, _ = Cenv(T, conj(T), C*C; ifsimple_eig=false)

    E = CTCtoT(C, T)
    Ǝ = CTCtoT(C, T)
    # 田 = sum(ein"abc,cba->"(FLmap(E, T, T, M), Ǝ))
    # 日 = sum(ein"abc,cba->"(E,Ǝ))
    田 = @tensor FLmap(E, T, T, M)[a,b,c] * Ǝ[c,b,a]
    日 = @tensor E[a,b,c] * Ǝ[c,b,a]
    λM = 田/日

    C = C * C
    Ɔ = C
    # 日 = sum(ein"ab,ba->"(Lmap(C, T, T), Ɔ))
    # 口 = sum(ein"ab,ba->"(C,Ɔ))
    日 = @tensor Lmap(C, T, T)[a,b] * Ɔ[b,a]
    口 = @tensor C[a,b] * Ɔ[b,a]
    λN = 日/口

    return log(abs(λM/λN))
end

function logZ(M::leg4, env::VUMPSEnv)
    @unpack AL, C, T = env

    # alternative implementation easier but slow 
    # λM, _ = Eenv(AL, conj(AL), M, A; ifsimple_eig=false)

    AC = ALCtoAC(AL, C)
    # 田 = sum(ein"abc,abc->"(FLmap(T, AC, conj(AC), M), T))
    # 日 = sum(ein"(abc,ad),(ce,dbe)->"(T,C,conj(C),T))
    田 = @tensor FLmap(T, AC, conj(AC), M)[a,b,c] * T[a,b,c]
    日 = @tensor (T[a,b,c] * C[a,d]) * (conj(C[c,e]) * T[d,b,e])
    λM = 田/日

    return log(abs(λM))
end
