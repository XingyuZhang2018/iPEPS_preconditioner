using Test
using LinearAlgebra
using Random

# We test arnoldi_eig standalone before integrating into iPEPS_preconditioner
# For now, include the function directly to test it in isolation.

include(joinpath(@__DIR__, "..", "src", "contraction", "arnoldi_eig.jl"))

@testset "arnoldi_eig" begin
    @testset "symmetric matrix dominant eigenpair" begin
        Random.seed!(42)
        n = 20
        # Build matrix with known dominant eigenvalue
        D_diag = collect(1.0:n)
        D_diag[end] = 100.0  # dominant eigenvalue = 100
        Q, _ = qr(randn(n, n))
        Q = Matrix(Q)
        A = Q * Diagonal(D_diag) * Q'

        f(v) = A * v
        v0 = randn(n)

        λ, v = arnoldi_eig(f, v0; krylov_dim=15, ifvalue=true)

        @test abs(λ) ≈ 100.0 atol=1e-10
        # v should be an eigenvector
        @test norm(f(v) - λ * v) / abs(λ) < 1e-10
    end

    @testset "large dense SPD matrix dominant eigenpair" begin
        Random.seed!(123)
        n = 30
        B = randn(n, n)
        A = B' * B + 50.0 * I  # symmetric positive definite, dominant eigenvalue clearly real

        f(v) = A * v
        v0 = randn(n)

        λ_arnoldi, v_arnoldi = arnoldi_eig(f, v0; krylov_dim=20, ifvalue=true)

        # Compare with eigen
        λs = eigvals(A)
        λ_dom = λs[argmax(abs.(λs))]

        @test λ_arnoldi ≈ λ_dom rtol=1e-8
    end

    @testset "small krylov_dim gives inexact result" begin
        Random.seed!(42)
        n = 20
        D_diag = collect(1.0:n)
        D_diag[end] = 100.0
        Q, _ = qr(randn(n, n))
        Q = Matrix(Q)
        A = Q * Diagonal(D_diag) * Q'

        f(v) = A * v
        v0 = randn(n)

        # With krylov_dim=2, result should be less accurate than krylov_dim=15
        _, v_small = arnoldi_eig(f, v0; krylov_dim=2)
        _, v_large = arnoldi_eig(f, v0; krylov_dim=15)

        # Both should approximate the dominant eigenvector, but v_large better
        true_v = Q[:, end]
        overlap_small = abs(dot(v_small, true_v))
        overlap_large = abs(dot(v_large, true_v))
        @test overlap_large >= overlap_small - 0.01  # large is at least as good
    end

    @testset "works with reshape'd arrays (like tensors)" begin
        Random.seed!(42)
        # Simulate the tensor case: v is a matrix, f maps matrix->matrix
        χ, D = 8, 3
        B = randn(χ * D, χ * D)
        A_full = B' * B + 50.0 * I  # symmetric positive definite, dominant eigenvalue real

        f(v) = reshape(A_full * vec(v), χ, D)
        v0 = randn(χ, D)

        λ, v = arnoldi_eig(f, v0; krylov_dim=20, ifvalue=true)

        @test abs(λ) > 40.0  # should be around 50
        # v should approximately satisfy eigenequation
        residual = norm(f(v) - λ * v) / abs(λ)
        @test residual < 1e-6
    end

    @testset "ifvalue=false returns zero eigenvalue" begin
        Random.seed!(42)
        n = 10
        A = Diagonal(collect(1.0:n))

        f(v) = A * v
        v0 = randn(n)

        λ, v = arnoldi_eig(f, v0; krylov_dim=5, ifvalue=false)

        @test λ == 0.0
        @test norm(v) ≈ 1.0  # eigenvector should still be normalized
    end
end
