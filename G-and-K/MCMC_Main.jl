using Distributed, SharedArrays
@everywhere using JLD2
addprocs(30)

@everywhere include("MCMC.jl")

@fetchfrom 2 InteractiveUtils.varinfo()

@everywhere @load "data.jld2" y0
@everywhere @load "cov.jld2" cov_proposal

R = SharedArray{Float64}(100000,4,50)

@distributed for i = 1:50
    init_theta = rand(Uniform(0,10),4)
    @time R[:,:,i] = MCMC(100000,y0,init_theta,cov_proposal)
end




