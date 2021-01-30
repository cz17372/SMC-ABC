using Distributed, SharedArrays
@everywhere using JLD2
addprocs(30)

@everywhere include("MCMC.jl")
@everywhere @load "data.jld2" y0
@everywhere @load "cov.jld2" cov_proposal

R = SharedArray{Float64}(100000,4,50)

@distributed for i = 1:50
    init_theta = rand(Uniform(0,10),4)
    @time R[:,:,i] = MCMC(100000,y0,init_theta,cov_proposal)
end



using FileIO
R = load("/Users/changzhang/OneDrive - University of Bristol/Desktop/Tim/MCMC_Result_20Obs.jld2")

R = R["R"]

t = 2
plot(R[:,t,1]);plot!(R[:,t,2]);plot!(R[:,t,3]);plot!(R[:,t,4]);plot!(R[:,t,5])


