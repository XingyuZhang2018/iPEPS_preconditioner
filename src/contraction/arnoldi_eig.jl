"""
    arnoldi_eig(f, v; krylov_dim=10, ifvalue=false)

GPU-friendly Arnoldi iteration for the dominant eigenpair.

Builds a `krylov_dim`-dimensional Krylov subspace {v, f(v), f²(v), ...} using
modified Gram-Schmidt orthogonalization, then solves the small projected eigenproblem.

All operations on `v` use BLAS-level ops (dot, axpy!, norm) that are efficient on GPU arrays.
The small Hessenberg eigenproblem (krylov_dim × krylov_dim) is solved on CPU.

Returns `(λ, v)` with the same interface as `simple_eig`.
- `λ` is the dominant eigenvalue (only computed if `ifvalue=true`, else 0.0)
- `v` is the corresponding eigenvector, normalized

Note: Assumes the dominant eigenvalue is real (valid for iPEPS transfer matrices).
Caller must apply `orth_for_ad(v)` for AD correctness (done by `eigsolve_dispatch`).

# Arguments
- `f`: linear operator v -> f(v)
- `v`: initial vector (can be any AbstractArray)
- `krylov_dim`: dimension of the Krylov subspace (controls accuracy vs cost)
- `ifvalue`: whether to compute the eigenvalue
"""
function arnoldi_eig(f, v; krylov_dim=10, ifvalue=false)
    T = real(eltype(v))

    # Normalize initial vector
    v = v / norm(v)

    # Storage for Krylov basis vectors
    Q = typeof(v)[]  # Vector of arrays, same type as v
    push!(Q, copy(v))

    # Upper Hessenberg matrix (on CPU, always small)
    H = zeros(T, krylov_dim + 1, krylov_dim)

    actual_dim = krylov_dim
    for j in 1:krylov_dim
        # Apply operator
        w = f(Q[j])

        # Modified Gram-Schmidt orthogonalization
        for i in 1:j
            h_ij = dot(Q[i], w)
            H[i, j] = h_ij
            w = w - h_ij * Q[i]
        end

        h_next = norm(w)
        H[j + 1, j] = h_next

        # Check for breakdown (lucky convergence)
        if h_next < eps(T) * 100
            # Krylov subspace is invariant, reduce dimension
            actual_dim = j
            break
        end

        if j < krylov_dim
            push!(Q, w / h_next)
        end
    end

    # Solve small eigenproblem on CPU
    H_k = Array(H[1:actual_dim, 1:actual_dim])
    eig_vals, eig_vecs = eigen(H_k)

    # Find dominant eigenvalue (largest magnitude)
    idx = argmax(abs.(eig_vals))
    # Sanity check: dominant eigenvalue should be real for transfer matrices
    if abs(imag(eig_vals[idx])) > 1e-6 * abs(eig_vals[idx])
        @warn "arnoldi_eig: dominant eigenvalue has significant imaginary part" eig_vals[idx]
    end
    λ = real(eig_vals[idx])
    y = eig_vecs[:, idx]

    # Reconstruct eigenvector in original space
    v_result = real(y[1]) * Q[1]
    for i in 2:min(length(y), length(Q))
        v_result = v_result + real(y[i]) * Q[i]
    end

    # Normalize
    v_result = v_result / norm(v_result)

    if !ifvalue
        λ = zero(T)
    end

    return λ, v_result
end
