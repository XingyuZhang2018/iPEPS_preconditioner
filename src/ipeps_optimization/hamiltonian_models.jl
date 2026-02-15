function spilt_h(h)
    d = size(h, 1)
    U, S, V = svd(reshape(h,d^2,d^2))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc])
    h2 = V[:,1:truc]'
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end

function hamiltonian_trunc(model)
    h = hamiltonian(model)
    d = size(h, 1)
    U, S, V = svd(reshape(h,d^2,d^2))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc]) 
    h2 = V[:,1:truc]'
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end

abstract type HamiltonianModel end

"""
    Heisenberg(Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
@kwdef mutable struct Heisenberg <: HamiltonianModel
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
end

const Sx = Float64[0 1; 1 0]/2
const Sy = ComplexF64[0 -1im; 1im 0]/2
const Sz = Float64[1 0; 0 -1]/2
const Sp = Float64[0 1; 0 0]
const Sm = Float64[0 0; 1 0]
"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    # h = model.Jx * ein"ij,kl -> ijkl"(Sx, Sx) +
    #     model.Jy * ein"ij,kl -> ijkl"(Sy, Sy) +
    #     model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
    # h = model.Jx * ein"ij,kl -> ijkl"(Sp, Sm) +
        # model.Jy/2 * ein"ij,kl -> ijkl"(Sm, Sp) +
        # model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
    # h = ein"ijcd,kc,ld -> ijkl"(h, Sx*2,(Sx*2)')
    h = model.Jx * (@tensor out[i,j,k,l] := Sp[i,j] * Sm[k,l]) +
        model.Jz * (@tensor out[i,j,k,l] := Sz[i,j] * Sz[k,l]) 
        
    @tensor h[j,i,l,k] = h[i,j,c,d] * (Sx*2)[k,c] * (Sx*2)[l,d]

    return real(h)
end
