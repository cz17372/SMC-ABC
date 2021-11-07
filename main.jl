using Distributions, StatsPlots, Plots, KernelDensity, Measures, Random, LinearAlgebra
theme(:ggplot2)

# Define Distance Metrics
Euclidean(x,y) = norm(x .- y)
Ordered(x,y)   = norm(sort(x).-sort(y))
# Import Samplers
include("src/Sampler/RESMC.jl") # Prangle's Algorithm, "RESMC"
include("src/Sampler/RWSMC.jl") # Novel Latents ABC-SMC
include("src/Sampler/ABCSMC.jl")# Del Moral's ABC-SMC

# g-and-k example
include("src/Model/gk/gku.jl")
include("src/Model/gk/gkn.jl")


Random.seed!(123)
θstar = [3.0,1.0,2.0,0.5]
ystar = gku.ConSimulator(20,θstar)

function SMC_CompCost(R)
    T = length(R.EPSILON) - 1
    Cost = 0
    for n = 1:T
        Cost += sum(R.WEIGHT[:,n+1] .> 0) * R.K[n]
    end
    return Cost/sum(R.WEIGHT[:,end] .> 0)
end


function gkResults(epsvec::Vector{Float64},M,N,ystar,Dist,θ0)
    k = length(epsvec)
    RWSMC_posterior = Array{Any,2}(undef,k,4)
    ABCSMC_posterior = Array{Any,2}(undef,k,4)
    RESMC_MCMC = Array{Matrix,1}(undef,k)
    RESMC_CompCost = zeros(k)
    RWSMC_CompCost = zeros(k)
    ABCSMC_CompCost = zeros(k)
    for n = 1:k
        # Perform the RWABCSMC
        R = RWSMC.SMC(N,ystar,gkn,Dist,η=0.8,TerminalTol=epsvec[n],TerminalProb=0.01)
        Index = findall(R.WEIGHT[:,end] .> 0)
        X     = gkn.GetPostSample(R)
        for j = 1:4
            RWSMC_posterior[n,j] = kde(X[j,:])
        end
        Σ = cov(X,dims=2)
        RWSMC_CompCost[n] = SMC_CompCost(R)
        R = ABCSMC.SMC(N,ystar,gkn,Dist,η=0.8,TerminalTol=epsvec[n],TerminalProb=0.0001)
        if R.EPSILON[end] == epsvec[n]
            Index = findall(R.WEIGHT[:,end] .> 0)
            X = R.U[end][:,Index]
            for j = 1:4
                ABCSMC_posterior[n,j] = kde(X[j,:])
            end
            ABCSMC_CompCost[n]=SMC_CompCost(R)
        else
            ABCSMC_CompCost[n] = Inf
        end
        R = RESMC.PMMH(θ0,M,N,y=ystar,model=gku,Dist=Dist,Σ=Σ,ϵ=epsvec[n])
        RESMC_MCMC[n] = R.theta
        RESMC_CompCost[n] = sum(R.NumVec)/M
    end
    return (Posterior = (RWABCSMC=RWSMC_posterior,ABCSMC=ABCSMC_posterior,RESMC=RESMC_MCMC),CompCost = (RWABCSMC=RWSMC_CompCost,ABCSMC=ABCSMC_CompCost,RESMC=RESMC_CompCost))
end

eps = [25.0,20.0,15,10,5,2,1,0.5,0.2]
gk_20data = gkResults(eps,2000,5000,ystar,Euclidean,θstar)


