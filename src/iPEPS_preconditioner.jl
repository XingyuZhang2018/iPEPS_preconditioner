module iPEPS_preconditioner

using AMDGPU
using CUDA
using ChainRulesCore
using KrylovKit
# using OMEinsum
using TensorOperations
using cuTENSOR
using OptimKit
using Parameters
using Zygote
using LinearAlgebra
using Printf
using Random
using JLD2
using FileIO
using ForwardDiff
using ForwardDiffChainRules
using MPI

import CUDA: CuArray
import Base: Array

export QRCTM, VUMPS
export environment
export hamiltonian, Heisenberg
export init_ipeps, optimise_ipeps, GradientOptimize, observable

include("defaults.jl")
include("contraction/utilities.jl")

include("contraction/unit_contraction/basic.jl")
include("contraction/unit_contraction/forloop_parallel_MPI.jl")
include("contraction/interface.jl")
include("contraction/environment.jl")
include("contraction/observable.jl")
include("contraction/runtime.jl")
include("contraction/autodiff.jl")

include("ipeps_optimization/interface.jl")
include("ipeps_optimization/init_ipeps.jl")
include("ipeps_optimization/restriction.jl")
include("ipeps_optimization/hamiltonian_models.jl")
include("ipeps_optimization/contraction.jl")
include("ipeps_optimization/energy_contraction.jl")
include("ipeps_optimization/observable.jl")
include("ipeps_optimization/precondition.jl")
include("ipeps_optimization/optimise_patch.jl")
include("ipeps_optimization/optimise_ipeps.jl")

end
