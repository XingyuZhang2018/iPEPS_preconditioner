abstract type Algorithm end

@with_kw mutable struct QRCTM <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    maxiter_ad::Int = Defaults.maxiter_ad
    miniter_ad::Int = Defaults.miniter_ad
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = Defaults.ifsimple_eig
    maxiter_power::Int = 1 # power steps for eigsolve
    ifload_env::Bool = true
    ifsave_env::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1 
end

@with_kw mutable struct VUMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    miniter::Int = Defaults.miniter
    maxiter_ad::Int = Defaults.maxiter_ad
    miniter_ad::Int = Defaults.miniter_ad
    output_interval::Int = 1
    verbosity::Int = Defaults.verbosity
    ifsimple_eig::Bool = Defaults.ifsimple_eig
    maxiter_power::Int = 1 # power steps for eigsolve
    maxiter_power_ad::Int = 5 # power steps for eigsolve in ad
    ifload_env::Bool = true
    ifsave_env::Bool = true
    ifparallel::Bool = false
    ifcheckpoint::Bool = false
    forloop_iter::Int = 1 
end

function environment(M, alg::Algorithm)
    env = initialize_env(M, alg)
    return environment(M, env, alg)
end