using Test
using iPEPS_preconditioner

@testset "eigsolver parameter defaults" begin
    alg_v = VUMPS()
    @test alg_v.eigsolver == :power
    @test alg_v.krylov_dim == 10
    @test alg_v.krylov_dim_ad == 10

    alg_q = QRCTM()
    @test alg_q.eigsolver == :power
    @test alg_q.krylov_dim == 10

    alg_old = VUMPS(ifsimple_eig=false)
    @test alg_old.eigsolver == :power
end

@testset "eigsolver=:arnoldi constructor" begin
    alg = VUMPS(eigsolver=:arnoldi, krylov_dim=5, krylov_dim_ad=8)
    @test alg.eigsolver == :arnoldi
    @test alg.krylov_dim == 5
    @test alg.krylov_dim_ad == 8
end
