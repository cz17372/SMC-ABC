using Distributions, Random, JLD2, LinearAlgebra, ProgressMeter
using Flux: gradient
using ForwardDiff: derivative

# Define the g-and-k function that transform standard Normal RV into the g-and-k RVs
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;


#                   Naive-SMC-ABC                   #

NaivelogPrior(θ) = sum(logpdf.(Uniform(0,10),θ))
NaiveDist(x) = norm(sort(x) .- sort(ystar))
function NaiveSMCABC_LocalMH(θ0,x0,ϵ; Σ, σ)
    #=
    Parameters:
    ---------------
    θ0: the current state of the parameters. This will be one of the particle from the previous iteration
    ϵ : the epsilon value for the current SMC step

    =#
    
    # Make a proposal based on random walk with mean θ0 and variance-covariance matrix σ^2*Σ

    newθ = rand(MultivariateNormal(θ0,σ^2*Σ))
    
    if NaivelogPrior(newθ) == -Inf
        # In this case, the proposal is not in the support of the prior, we will straightaway 
        # reject the proposals
        return (θ0,x0,0)
    else
        # In the case where the proposal lies in the support of the prior
        # 1. We sample a new set of data based on newθ
        z = rand(Normal(0,1),length(x0))
        newx = f.(z,θ=newθ)
        if NaiveDist(newx) < ϵ
            return (newθ,newx,1)
        else
            return (θ0,x0,0)
        end
    end
end
function NaiveSMCABC(N,T,NoData;Threshold,σ,λ)
    THETA = zeros(4,N,T+1)
    EPSILON = zeros(T+1)
    X = zeros(NoData,N,T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    SIGMA = zeros(T+1); SIGMA[1] = σ
    ACCEPTANCE = zeros(T)
    # Create the initial particles from prior
    for i = 1:N
        THETA[:,i,1] = rand(Uniform(0,10),4);
        z = rand(Normal(0,1),NoData)
        X[:,i,1] = f.(z,θ=THETA[:,i,1])
        DISTANCE[i,1] = NaiveDist(X[:,i,1]) 
    end

    # Set the initial ϵ to be the largest distance among the initial particles
    EPSILON[1],_ = findmax(DISTANCE[:,1])
    WEIGHT[:,1] .= 1/N

    acc_vec = zeros(N)
    @showprogress 1 "Computing..." for t = 1:T
        # We first do the resampling and determine the next ϵ value based on the resampled particles
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        # To have a ESS of at least Threshod*N we find the "Threshold" quantile of the distances
        EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .<= EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .<= EPSILON[t+1])
        print(EPSILON[t+1],",",length(findall(WEIGHT[:,t+1] .> 0)),"\n")
        Σ = cov(THETA[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            THETA[:,i,t+1], X[:,i,t+1],acc_vec[i] = NaiveSMCABC_LocalMH(THETA[:,ANCESTOR[i,t],t],X[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ = SIGMA[t])
            DISTANCE[i,t+1] = NaiveDist(X[:,i,t+1])
        end
        ACCEPTANCE[t] = mean(acc_vec)
        SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(ACCEPTANCE[t]-0.234))
    end
    return (THETA=THETA,X=X,WEIGHT=WEIGHT,EPSILON=EPSILON,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,ANCESTOR=ANCESTOR,DISTANCE=DISTANCE)
end
