module Defaults
    const VERBOSE_NONE = 0
    const VERBOSE_WARN = 1
    const VERBOSE_CONV = 2
    const VERBOSE_ITER = 3
    const VERBOSE_ALL = 4

    using OptimKit
    using KrylovKit
    const verbosity = VERBOSE_ITER
    const maxiter = 1000
    const miniter = 0
    const maxiter_ad = 100
    const miniter_ad = 50
    const tol = 1e-12
    const ifcheckpoint = false
    const ifsimple_eig = true
    
    const fpgrad_maxiter = 100
    const fpgrad_tol = 1e-6
    const output_interval = 1
    const save_interval = 1
    const folder = "data"
    const reuse_env = true
    const rrule_alg = GMRES(; tol=tol)
    const optimizer = LBFGS(; verbosity = 0)
end