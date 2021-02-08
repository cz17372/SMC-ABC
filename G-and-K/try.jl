using Flux:gradient
using ProgressMeter, Distributions, LinearAlgebra
using JLD2

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

@load "/Users/changzhang/Documents/GitHub/SMC-ABC/G-and-K/data.jld2" y0

ystar = y0;

Langevin_logPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4]))+sum(logpdf.(Normal(0,1),ξ[5:end]))

Langevin_Dist(ξ) = norm(f.(ξ[5:end],θ=ξ[1:4]) .- ystar)
grad(ξ) = normalize(gradient(Langevin_Dist,ξ)[1])
function Langevin_SMC_ABC_LocalMH(ξ0,ϵ;Σ,σ)
    newξ = rand(MultivariateNormal(ξ0 .- σ/2*Σ*grad(ξ0),σ*Σ))
    forward_proposal_density = logpdf(MultivariateNormal(ξ0 .- σ/2*Σ*grad(ξ0),σ*Σ),newξ)
    backward_proposal_density = logpdf(MultivariateNormal(newξ .- σ/2*Σ* grad(newξ),σ*Σ),ξ0)
    log_propose_ratio = backward_proposal_density - forward_proposal_density
    logPrior_ratio = Langevin_logPrior(newξ)-Langevin_logPrior(ξ0)
    u = log(rand(Uniform(0,1)))
    if u >= logPrior_ratio + log_propose_ratio
        return (ξ0,0,log_propose_ratio,logPrior_ratio)
    else
        if Langevin_Dist(newξ) < ϵ
            return (newξ,1,log_propose_ratio,logPrior_ratio)
        else
            return (newξ,0,log_propose_ratio,logPrior_ratio)
        end
    end
end

function Langevin_SMC_ABC(N,T,NoData,Threshold,σ,λ)
    XI       = zeros(4+NoData,N,T+1);
    EPSILON  = zeros(T+1);
    DISTANCE = zeros(N,T+1);
    WEIGHT   = zeros(N,T+1);
    ANCESTOR = zeros(Int,N,T);
    LOGPROPOSAL = zeros(N,T);
    LOGPRIOR = zeros(N,T)
    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = Langevin_Dist(XI[:,i,1])
    end
    SIGMA    = zeros(T+1)
    SIGMA[1] = σ
    ACCEPTANCE = zeros(T)
    WEIGHT[:,1] .= 1/N;
    EPSILON[1]  = findmax(DISTANCE[:,1])[1]
    
    @showprogress 1 "Computing.." for t = 1:T
        accepted = 0
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[:,t]))<Int(Threshold*N)
            EPSILON[t+1] = EPSILON[t]
        else
            EPSILON[t+1] = sort(unique(DISTANCE[:,t]))[Int(Threshold*N)]
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            newξ,dec,logprop,logprior= Langevin_SMC_ABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t])
            accepted += dec
            XI[:,i,t+1] = newξ;
            LOGPROPOSAL[i,t] = logprop
            LOGPRIOR[i,t]    = logprior
            DISTANCE[i,t+1] = Langevin_Dist(XI[:,i,t+1])
        end
        SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(accepted/N-0.45))
        ACCEPTANCE[t] = accepted/N
    end
    return (XI=XI,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,LOGPRIOR=LOGPRIOR,LOGPROPOSAL=LOGPROPOSAL)
end
