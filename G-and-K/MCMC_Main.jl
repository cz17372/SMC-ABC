using Distributed, SharedArrays
addprocs(4)

include("MCMC.jl")
