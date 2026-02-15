abstract type iPEPSOptimize end

@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    boundary_alg::Algorithm
    iffixedpoint::Bool = false
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    optimizer = Defaults.optimizer
    folder::String = Defaults.folder
    output_interval::Int = Defaults.output_interval
    save_interval::Int = Defaults.save_interval
    ifsave_lbfgs::Bool = true
    ifload_lbfgs::Bool = true
    
    preconditiontype = :local
    iter_precond::Int = 10
end