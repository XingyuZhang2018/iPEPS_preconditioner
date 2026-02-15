using Test
using Random
using iPEPS_preconditioner

function run_short_example()
    Random.seed!(42)
    atype = Array
    D, χ, χshift = 2, 10, 0
    model = Heisenberg()
    folder = folder = joinpath(pkgdir(iPEPS_preconditioner), "data/$model/QRCTM/")

    boundary_alg = QRCTM(maxiter=3,
                         maxiter_ad=20,
                         miniter_ad=20,
                         tol=1e-10,
                         verbosity=0,
                         output_interval=10,
                         ifcheckpoint=false,
                         ifload_env=false,
                         forloop_iter=1)

    params = GradientOptimize(boundary_alg=boundary_alg,
                              iffixedpoint=false,
                              optimizer=LBFGS(10; maxiter=100, verbosity=0, gradtol=1e-6),
                              reuse_env=false,
                              verbosity=3,
                              folder=folder,
                              preconditiontype=:none,
                              ifload_lbfgs=false,
                              iter_precond=0,
                              save_interval=1)

    A = init_ipeps(; atype, No=0, d=2, D, χ, params)

    # The test verifies the example runs without throwing an error.
    optimise_ipeps(A, χ, χshift, model, params)
    return true
end

@testset "Heisenberg QRCTM example" begin
    @test run_short_example()
end
