function precondition_inverse_local_metric(A, grad, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    t0 = time()
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    
    @unpack C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    To = CTCtoT(C, T)

    gradnew, _ = linsolve(grad; isposdef = true, maxiter=1, verbosity=0) do x
        return δ * x + Mumap_parallel(T, T, To, To, x; forloop_iter, ifparallel)
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    
    return gradnew
end

function environment_FWAD(M, env::CTMEnv, alg::Algorithm)
    err = Inf
    for i = 1:alg.maxiter
        env = leftmove(ForwardDiff.value.(M), env, alg) 
    end

    for i = 1:alg.maxiter_ad
        env = leftmove(M, env, alg) 
    end

    return env, err
end

function precondition_inverse_full_metric(A, g, env::CTMEnv, params, restriction_ipeps, fδEi, iter_precond)
    t0 = time()
    if fδEi[3] <= iter_precond
        return g
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)

    algr = deepcopy(params.boundary_alg)
    algr.verbosity = 0
    gnew, _ = linsolve(g; isposdef = true, maxiter=1, verbosity=0) do x
        function f(Ad)
            env, _ = environment((A, Ad), env, algr)
            @unpack C, T = env
            @unpack forloop_iter, ifparallel = algr
            To = CTCtoT(C, T)

            return sum((@tensor FLmap_parallel(To, T, T, (x, Ad); ifparallel, forloop_iter)[a,b,c,d] * To[a,b,c,d]))
        end
        gN = Zygote.gradient(f, A)[1]

        # function f(Au)
        #     env, _ = environment_FWAD((restriction_ipeps(Au), restriction_ipeps(A)), env, algr)
        #     # Zygote.@ignore update!(env, env′)
        #     @unpack C, T = env
        #     @unpack forloop_iter, ifparallel = algr
        #     To = CTCtoT(C, T)

        #     # return Mumap_parallel(T, T, To, To, Au; forloop_iter, ifparallel)
        #     # ForwardDiff.gradient(y->sum(ein"abcdp,abcdp->"(Mumap_parallel(T, T, To, To, restriction_ipeps(Au); forloop_iter, ifparallel), conj(restriction_ipeps(y)))), A)
        #     # ForwardDiff.gradient(y -> (@tensor Mumap_parallel(T, T, To, To, restriction_ipeps(Au); forloop_iter, ifparallel)[a,b,c,d,p] * conj(restriction_ipeps(y))[a,b,c,d,p]), A)
        #     return sum((@tensor FLmap_parallel(To, T, T, (x, Ad); ifparallel, forloop_iter)[a,b,c,d], To[a,b,c,d]))
        # end
        # gN = ForwardDiff.derivative(t -> f(A + t * x), 0.0)
        return δ * x + gN
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    
    return gnew
end


function precondition_inverse_local_metric(A, grad, env::VUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    t0 = time()
    δ = fδEi[2]
    A = restriction_ipeps(A)

    @unpack AL, C, T = env
    @unpack forloop_iter, ifparallel = params.boundary_alg
    AC = ALCtoAC(AL,C)

    gradnew, _ = linsolve(grad; isposdef = true, maxiter=1, verbosity=0) do x
        return δ * x + Mumap_parallel(AC, AC, T, T, x; forloop_iter, ifparallel)
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    return gradnew
end

function precondition_inverse_full_metric(A, g, env::VUMPSEnv, params, restriction_ipeps, fδEi, iter_precond)
    t0 = time()
    if fδEi[3] <= iter_precond
        return g
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)

    algr = deepcopy(params.boundary_alg)
    algr.verbosity = 0
    gnew, _ = linsolve(g; isposdef = true, maxiter=1, verbosity=0) do x
        function f(Ad)
            env, _ = environment((A, Ad), env, algr)
            @unpack AL, C, T = env
            @unpack forloop_iter, ifparallel = algr
            AC = ALCtoAC(AL,C)

            return sum((@tensor FLmap_parallel(T, AC, AC, (x, Ad); ifparallel, forloop_iter)[a,b,c,d] * T[a,b,c,d]))
        end
        gN = Zygote.gradient(f, A)[1]
        return δ * x + gN
    end

    params.verbosity >= 2 && printstyled("precondition calculation took $(round(time() - t0, digits = 2)) s\n"; bold=true, color=:green)
    
    return gnew
end

function precondition_inverse_BP(A, grad, restriction_ipeps, fδEi, iter_precond)
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    D = size(A, 1)

    B = _arraytype(M)(rand(eltype(M), D, D))
    
    error = 1.0
    Z = 1.0
    for i in 1:100
        B[c,g] = A[a,b,c,d,p] * B[a,e] * B[b,f] * B[d,h] * conj(A[e,f,g,h,p])
        Z_n = dot(B,B)
        normalize!(B)
        error = norm(Z_n - Z)
        if error < 1e-16
            break
        end
        Z = Z_n
    end
    gradnew = deepcopy(grad)

    # n = sum(ein"((abcd,d),c),b,a ->"(M,B,B,B,B))
    n = @tensor A[a,b,c,d,p] * B[a,e] * B[b,f] * B[d,h] * B[c,g] * conj(A[e,f,g,h,p])
    # gradnew, _ = linsolve(x->δ * x + ein"(((abcdp,ae),bf),cg),dh->efghp"(x,reB,reB,reB,reB)/n, grad; isposdef = true, maxiter=1)
    gradnew, _ = linsolve(x->δ * x + (@tensor out[e,f,g,h,p] := x[a,b,c,d,p] * B[a,e] * B[b,f] * B[c,g] * B[d,h])/n, grad; isposdef = true, maxiter=1)

    return gradnew
end