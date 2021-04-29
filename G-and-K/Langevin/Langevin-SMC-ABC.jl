module Langevin
using Distributions, LinearAlgebra
using ForwardDiff: gradient

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

Dist(ξ;y) = norm(f.(ξ[5:end],θ=ξ[1:4]) .- y)

logPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4])) + sum(logpdf.(Normal(0,1),ξ[5:end]))

function LSMCABC_LocalMH(N,ξ0,ϵ;Σ,σ,y) 
    ξ = zeros(N,length(ξ0))
    ξ[1,:] = ξ0
    AcceptedNum = 0
    for n = 2:N
        μ = ξ[n-1,:] .- σ^2/2*Σ*normalize(gradient(x->Dist(x,y=y),ξ[n-1,:]))
        newξ = rand(MultivariateNormal(μ,σ^2*Σ))
        reverseμ = newξ .- σ^2/2 * Σ * normalize(gradient(x-> Dist(x,y=y),ξ[n-1,:]))
        forward_proposal_density = logpdf(MultivariateNormal(μ,σ^2*Σ),newξ)
        backward_proposal_density = logpdf(MultivariateNormal(reverseμ,σ^2*Σ),ξ[n-1,:])
        log_proposal_ratio = backward_proposal_density-forward_proposal_density
        log_prior_ratio = logPrior(newξ) - logPrior(ξ[n-1,:])
        u = log(rand(Uniform(0,1)))
        if u > log_prior_ratio + log_proposal_ratio
            ξ[n,:] = ξ[n-1,:]
        else
            if Dist(newξ,y=y) < ϵ
                ξ[n,:] = newξ
                AcceptedNum += 1
            else
                ξ[n,:] = ξ[n-1,:]
            end
        end
    end
    return ξ[end,:],AcceptedNum
end

function Langevin_SMC_ABC(N,T,NoData;y,Threshold,σ,K0)
    XI = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    K = zeros(Int64,T+1)
    K[1] = K0
    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = Dist(XI[:,i,1],y=y)
    end
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    ParticleAcceptProb = zeros(N)
    MH_AcceptProb = zeros(T)
    for t = 1:T
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        println("SMC Step: ", t)
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("K = ", K[t])
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        # L = cholesky(Σ).L
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            XI[:,index[i],t+1],ParticleAcceptProb[index[i]] = LSMCABC_LocalMH(K[t],XI[:,ANCESTOR[index[i],t],t],EPSILON[t+1],Σ=Σ,σ=σ,y=y)
            DISTANCE[index[i],t+1] = Dist(XI[:,index[i],t+1],y=y)
        end
        MH_AcceptProb[t] = mean(ParticleAcceptProb[index])/K[t]
        K[t+1] = Int64(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        print("\n\n")
    end
    return (XI=XI,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K)
end

end
