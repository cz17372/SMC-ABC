using Distributions, Random, JLD2, LinearAlgebra, ProgressMeter
using Flux: gradient
using ForwardDiff: derivative

# Define the g-and-k function that transform standard Normal RV into the g-and-k RVs
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function summary(y)
    E = quantile(y,collect(1/8:1/8:1))
    S = zeros(4)
    S[1] = E[4]; S[2] = E[6]-E[2]; S[3] = (E[6]+E[2]-2*E[4])/S[2]
    S[4] = (E[7]-E[5]+E[3]-E[1])/S[2]
    return S
end

#                   Naive-SMC-ABC                                   #

NaivelogPrior(θ) = sum(logpdf.(Uniform(0,10),θ))
NaiveDist(x) = norm(summary(x) .- summary(ystar))

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
function NaiveSMCABC(N,T,NoData;Threshold,σ,λ,Method="ESS")
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
    for t = 1:T
        print("Iteration ",t,"\n")
        # We first do the resampling and determine the next ϵ value based on the resampled particles
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        # To have a ESS of at least Threshod*N we find the "Threshold" quantile of the distances
        #=
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        elseif Method == "Unique"
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) >= Threshold
                EPSILON[t+1] = sort(unique(DISTANCE[ANCESTOR[:,t],t]))[Threshold]
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end
        =#
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        else
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) > 4000
                EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end

        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .<= EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .<= EPSILON[t+1])
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


#                   Random-Walk SMC-ABC (RW-SMC-ABC)                #

RWlogPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4])) + sum(logpdf.(Normal(0,1),ξ[5:end]))
RWDist(ξ)     = norm(summary(f.(ξ[5:end],θ=ξ[1:4])) .- summary(ystar))
function RWSMCABC_LocalMH(ξ0,ϵ;Σ,σ)
    # Propose a new particle
    newξ = rand(MultivariateNormal(ξ0,σ^2*Σ))
    u = log(rand(Uniform(0,1)))
    if u > (RWlogPrior(newξ)-RWlogPrior(ξ0))
        # Early rejection
        return (ξ0,0)
    else
        if RWDist(newξ) < ϵ
            return (newξ,1)
        else
            return (ξ0,0)
        end
    end
end
function RWSMCABC(N,T,NoData;Threshold,σ,λ,Method="ESS")
    XI = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = RWDist(XI[:,i,1])
    end

    SIGMA = zeros(T+1)
    SIGMA[1] = σ
    ACCEPTANCE = zeros(T)
    WEIGHT[:,1] .= 1/N

    EPSILON[1],_ = findmax(DISTANCE[:,1])

    accepted = zeros(N)

    @showprogress 1 "Computing.." for t = 1:T
        #print("Iteration ",t,"\n")
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        #=
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        elseif Method == "Unique"
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) >= Threshold
                EPSILON[t+1] = sort(unique(DISTANCE[ANCESTOR[:,t],t]))[Threshold]
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end
        =#
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        else
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) > 4000
                EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            XI[:,i,t+1],accepted[i] = RWSMCABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t])
            DISTANCE[i,t+1] = RWDist(XI[:,i,t+1])
        end
        ACCEPTANCE[t] = mean(accepted)
        SIGMA[t+1] = exp(log(SIGMA[t])+λ*(ACCEPTANCE[t]-0.234))
    end
    return (XI=XI,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,ANCESTOR=ANCESTOR)
end



#                   Langevin SMC-ABC (L-SMC-ABC)                    #

LlogPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4])) + sum(logpdf.(Normal(0,1),ξ[5:end]))
LDist(ξ) = norm(summary(f.(ξ[5:end],θ=ξ[1:4])) .- summary(ystar))
gradDist(ξ) = norm(f.(ξ[5:end],θ=ξ[1:4]) .- ystar)^2
#=
function grad(ξ)
    gd = gradient(gradDist,ξ)[1]
    if norm(gd) > 5
        return normalize(gd)
    else
        return gd
    end
end
=#

grad(ξ)   = normalize(gradient(gradDist,ξ)[1])

function LSMCABC_LocalMH(ξ0,ϵ;Σ,σ)
    μ = ξ0 .- σ^2/2 * Σ * grad(ξ0) 
    newξ = rand(MultivariateNormal(μ,σ^2*Σ))

    reverseμ = newξ .- σ^2/2 * Σ * grad(newξ)
    
    forward_proposal_density = logpdf(MultivariateNormal(ξ0,σ^2*Σ),newξ)
    backward_proposal_density = logpdf(MultivariateNormal(reverseμ,σ^2*Σ),ξ0)

    log_proposal_ratio = backward_proposal_density-forward_proposal_density

    log_prior_ratio = LlogPrior(newξ) - LlogPrior(ξ0)

    u = log(rand(Uniform(0,1)))
    
    if u >= log_proposal_ratio + log_prior_ratio
        return (ξ0,0)
    else
        if LDist(newξ) < ϵ
            return (newξ,1)
        else
            return (ξ0,0)
        end
    end
end
function LSMCABC(N,T,NoData;Threshold,σ,λ,Method="ESS")
    XI = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)

    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = LDist(XI[:,i,1])
    end

    SIGMA = zeros(T+1)
    SIGMA[1] = σ

    ACCEPTANCE = zeros(T)
    WEIGHT[:,1] .= 1/N
    EPSILON[1],_ = findmax(DISTANCE[:,1])
    accepted = zeros(N)
    for t = 1:T
        print("Iteration ",t,"\n")
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...)
        #=
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        elseif Method == "Unique"
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) >= Threshold
                EPSILON[t+1] = sort(unique(DISTANCE[ANCESTOR[:,t],t]))[Threshold]
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end
        =#
        if Method == "ESS"
            EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        else
            if length(unique(DISTANCE[ANCESTOR[:,t],t])) > 4000
                EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
            else
                EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
            end
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            XI[:,i,t+1],accepted[i] = LSMCABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t])
            DISTANCE[i,t+1] = LDist(XI[:,i,t+1])
        end
        ACCEPTANCE[t] = mean(accepted)
        SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(ACCEPTANCE[t] - 0.45) )
    end
    return (XI=XI,DISTANCE=DISTANCE,EPSILON=EPSILON,WEIGHT=WEIGHT,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,ANCESTOR=ANCESTOR)
end


