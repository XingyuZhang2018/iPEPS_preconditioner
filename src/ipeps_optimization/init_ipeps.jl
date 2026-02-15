function init_ipeps(;atype = Array, No, d::Int, D::Int, χ::Int, params)
    if No != 0
        file = joinpath("$(params.folder)","D$(D)","ipeps","χ$(χ)","No.$(No).jld2")
        A = load(file, "bcipeps")
        params.verbosity >= 2 && @info "load ipeps from $file"
    else
        A = ones(Float64, D,D,D,D,d) + rand(Float64, D,D,D,D,d)
        A /= norm(A)
        params.verbosity >= 2 && @info "random initial ipeps"
    end
    return atype(A)
end

function init_ipeps_from_small_D(;atype = Array, No, D::Int, D′::Int, d::Int, χ::Int, ϵ::Float64=1e-3, params)
    D < D′ || throw(Base.error(" D thould smaller than D′"))
    file = joinpath("$(params.folder)","D$(D)","ipeps","χ$(χ)","No.$(No).jld2")
    A = load(file, "bcipeps")
    A′ = ϵ * rand(Float64, D′,D′,D′,D′,d)
    A′[1:D, 1:D, 1:D, 1:D, :] = A
    params.verbosity >= 2 && @info "load ipeps from $file, enlarge to $D′"
    return atype(A′)
end