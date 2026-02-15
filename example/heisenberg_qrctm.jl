using iPEPS_preconditioner
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

for seed in 72:72, preconditiontype in [:none, :local, :full]
    Random.seed!(seed)
    atype = Array #or CuArray
    D, χ, χshift = 2, 10, 0
    model = Heisenberg()
    No = 0
    folder = joinpath(pkgdir(iPEPS_preconditioner), "data/$model/QRCTM/seed$seed/$preconditiontype")
    boundary_alg = QRCTM(maxiter=3,
                         maxiter_ad=20,
                         miniter_ad=20,
                         tol=1e-10,
                         verbosity=2,
                         output_interval=10,
                         ifcheckpoint=false,
                         ifload_env=true,
                         forloop_iter=1
    )
    params = GradientOptimize(boundary_alg=boundary_alg, 
                              iffixedpoint=false,
                              optimizer=LBFGS(200; maxiter=1000, verbosity=4, gradtol=1e-9),
                              # optimizer=GradientDescent(maxiter=200, gradtol=1e-10),
                              reuse_env=true, 
                              verbosity=4, 
                              folder=folder,
                              preconditiontype=preconditiontype,
                              ifload_lbfgs=false,
                              iter_precond=0,
    )
    A = init_ipeps(;atype, No, d=2, D, χ, params)

    optimise_ipeps(A, χ, χshift, model, params);
end