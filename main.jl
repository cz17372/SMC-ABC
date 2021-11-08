using Distributions, StatsPlots, Plots, KernelDensity, Measures, Random, LinearAlgebra
theme(:ggplot2)

# Define Distance Metrics
Euclidean(x,y) = norm(x .- y)
Ordered(x,y)   = norm(sort(x).-sort(y))
# Import Samplers
include("src/Sampler/RESMC.jl") # Prangle's Algorithm, "RESMC"
include("src/Sampler/RWSMC.jl") # Novel Latents ABC-SMC
include("src/Sampler/ABCSMC.jl")
include("src/Sampler/MCMC.jl")# Del Moral's ABC-SMC

# g-and-k example
include("src/Model/gk/gku.jl")
include("src/Model/gk/gkn.jl")


Random.seed!(123)
θstar = [3.0,1.0,2.0,0.5]
ystar20 = gku.ConSimulator(20,θstar)

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
        if R == "Infeasible"
            RESMC_CompCost[n] = Inf
        else
            RESMC_MCMC[n] = R.theta
            RESMC_CompCost[n] = sum(R.NumVec)/M
        end
    end
    return (Posterior = (RWABCSMC=RWSMC_posterior,ABCSMC=ABCSMC_posterior,RESMC=RESMC_MCMC),CompCost = (RWABCSMC=RWSMC_CompCost,ABCSMC=ABCSMC_CompCost,RESMC=RESMC_CompCost))
end

#eps = [25.0,20.0,15,10,5,2,1,0.5,0.2]
#gk_20data = gkResults([25.0],2000,5000,ystar20,Euclidean,θstar)



Random.seed!(4013)
θstar = [3.0,1.0,2.0,0.5]
ystar250 = gku.ConSimulator(250,θstar)
RWSMC250data = RWSMC.SMC(5000,ystar250,gkn,Euclidean,η = 0.8, TerminalTol=0.5)

Random.seed!(17382)
θstar = [3.0,1.0,2.0,0.5]
ystar100 = gku.ConSimulator(100,θstar)
RWSMC100data = RWSMC.SMC(5000,ystar100,gkn,Euclidean,η = 0.8, TerminalTol=0.5)


Random.seed!(2021)
θstar = [3.0,1.0,2.0,0.5]
ystar50 = gku.ConSimulator(50,θstar)
RWSMC50data = RWSMC.SMC(5000,ystar50,gkn,Euclidean,η = 0.8, TerminalTol=0.5)

dat50 = Array{Any,1}(undef,4)
dat100 = Array{Any,1}(undef,4)
dat250 = Array{Any,1}(undef,4)

for i = 1:4
    dat50[i] = kde(gkn.GetPostSample(RWSMC50data)[i,:])
    dat100[i] = kde(gkn.GetPostSample(RWSMC100data)[i,:])
    dat250[i] = kde(gkn.GetPostSample(RWSMC250data)[i,:])
end

gkRWSMCPosterior = (data50=dat50,data100=dat100,data250=dat250)

MCMC50= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC50data),dims=2),0.2,y=ystar50)
MCMC100= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC100data),dims=2),0.4,y=ystar100)
MCMC250= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC250data),dims=2),1.0,y=ystar250)

MCMCposterior = (data50=MCMC50.Sample,data100=MCMC100.Sample,data250=MCMC250.Sample)

