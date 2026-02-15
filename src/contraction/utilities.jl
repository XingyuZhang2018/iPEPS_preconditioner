_arraytype(::CuArray) = CuArray
_arraytype(::Array) = Array

const leg3 = AbstractArray{T, 3} where T
const leg4 = AbstractArray{T, 4} where T
const leg5 = AbstractArray{T, 5} where T


function _to_front(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

function simple_eig(f, v; maxiter, ifvalue=false)
    λ = 0.0
    # Zygote.@ignore begin
    #     for _ in 1:20
    #         v = f(v)
    #         λ′ = norm(v)
    #         v /= λ′
    #         abs(λ′ - λ) < 1e-8 && break
    #         λ = λ′
    #     end
    # end
    # v = Zygote.@ignore v
    for _ in 1:maxiter
        v = f(v)
        v /= Zygote.@ignore norm(v)
    end

    v = orth_for_ad(v)
    if ifvalue
        λ = dot(v, f(v))
    end
    return λ, v
end

function qr_for_ad(A::AbstractMatrix{T}) where {T}
    Q, R = qr(A)
    Q = _arraytype(A)(Q)
    return Q, R
end

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = _arraytype(A)(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    Q .= Q * Diagonal(phases)
    R .= Diagonal(conj.(phases)) * R
    return Q, R
end

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...; kwargs...) = f(x...; kwargs...) 
Zygote.@adjoint checkpoint(f, args...; kwargs...) = f(args...; kwargs...), ȳ -> Zygote._pullback((args...) -> f(args...; kwargs...), args...)[2](ȳ)


function reclaim(::CuArray)
    if CUDA.available_memory() / CUDA.total_memory() < 0.1
        @warn raw"Low GPU memory, running garbage collection and reclaiming CUDA memory."
        GC.gc(true)
        CUDA.reclaim()
    end
end

function reclaim(::AbstractArray)
    return nothing
end

function synchronize(x::AbstractArray)
    if x isa CuArray
        CUDA.synchronize()
    elseif x isa ROCArray
        AMDGPU.synchronize()
    end
end

function save_env(file::String, env)
    env_save = Array(env)
    save(file, "env", env_save)
end

function load_env(file::String, atype)
    env = atype(load(file, "env"))
    return env
end