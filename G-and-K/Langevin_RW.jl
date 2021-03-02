using Distributions, Random
using Flux: gradient
using ForwardDiff: derivative
using ProgressMeter
using LinearAlgebra
using JLD2

# Load the data
@load "G-and-K/data.jld2" y0; ystar = y0;

# Transformation function
f(z;θ) =  θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;


#--------------------------------- Randwom-Walk SMC-ABC ------------------------------------#
RW_logPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4])) + sum(logpdf.(Normal(0,1),ξ[5:end]));
RW_Dist(ξ) = norm(sort(f.(ξ[5:end],θ=ξ[1:4])) .- sort(ystar));


function RW_SMC_ABC_LocalMH(ξ0,ϵ;Σ,σ,K)
    #=
    ξ0: starting point of the local Metropolis-Hastings exploration, this will be the particle from previous iteration
    ϵ:  the epsilon values for the current iteration, i.e. the target will be p(ξ|y) = p(ξ)l(x|ξ)I(d(y,x)<ϵ)
    Σ:  Covariance Matrix used to propose new values
    σ:  scale factors of the proposals
    K:  Number of MH steps for the local exploration
    =#
    accepted = 0
    C = size(Σ)[1]
    for i = 1:K
        xi_proposal = ξ0 .+  σ*rand(MultivariateNormal(zeros(C),Σ))
        u = log(rand(Uniform(0,1)))
        if u < (RW_logPrior(xi_proposal)-RW_logPrior(ξ0)) # An early rejection algorithm is employed here
            if RW_Dist(xi_proposal) < ϵ
                ξ0 = xi_proposal
                accepted += 1
            end
        end
    end
    return (ξ0,accepted)
end

function RW_SMC_ABC(N,T,NoData;Threshold,σ,λ,K)
    XI       = zeros(4+NoData,N,T+1);
    EPSILON  = zeros(T+1);
    DISTANCE = zeros(N,T+1);
    WEIGHT   = zeros(N,T+1);
    ANCESTOR = zeros(Int,N,T);
    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = RW_Dist(XI[:,i,1])
    end
    SIGMA    = zeros(T+1)
    SIGMA[1] = σ
    ACCEPTANCE = zeros(T)
    WEIGHT[:,1] .= 1/N;
    NewValues = zeros(T)
    EPSILON[1]  = findmax(DISTANCE[:,1])[1]
    accepted = zeros(N)
    @showprogress 1 "Computing.." for t = 1:T
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        #=
        if length(unique(DISTANCE[:,t]))<Int(Threshold*N)
            EPSILON[t+1] = EPSILON[t]
        else
            EPSILON[t+1] = sort(unique(DISTANCE[:,t]))[Int(Threshold*N)]
        end
        =#
        EPSILON[t+1] = quantile(DISTANCE[ANCESTOR[:,t],t],Threshold)
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            XI[:,i,t+1],accepted[i] = RW_SMC_ABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t],K=K)
            DISTANCE[i,t+1] = RW_Dist(XI[:,i,t+1])
        end
        SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(sum(accepted)/(K*N)-0.234))
        ACCEPTANCE[t] = sum(accepted)/(K*N)
        NewValues[t] = count(accepted .> 0)
    end
    return (XI=XI,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,NewValues=NewValues,ANCESTOR=ANCESTOR)
end

#=
function RW_SMC_ABC(N,T,NoData;Threshold=[0.3,0.5],CoolingSchedule=0.5,σ,λ,K)
    XI       = zeros(4+NoData,N,T+1);
    EPSILON  = zeros(T+1);
    DISTANCE = zeros(N,T+1);
    WEIGHT   = zeros(N,T+1);
    ANCESTOR = zeros(Int,N,T);
    for i = 1:N
        XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = RW_Dist(XI[:,i,1])
    end
    error_deduction = zeros(T+1)
    SIGMA    = zeros(T+1)
    SIGMA[1] = σ
    ACCEPTANCE = zeros(T)
    WEIGHT[:,1] .= 1/N;
    EPSILON[1]  = findmax(DISTANCE[:,1])[1]
    error_deduction[1] = EPSILON[1]
    @showprogress 1 "Computing..." for t = 1:T
        error_deduction[t+1] = min(error_deduction[t],CoolingSchedule*EPSILON[t])
        EPSILON[t+1] = EPSILON[t] - error_deduction[t+1]
        ANCESTOR[:,t] = collect(1:N)
        for i = 1:N
            WEIGHT[i,t+1] = WEIGHT[i,t] * (RW_Dist(XI[:,ANCESTOR[i,t],t])<EPSILON[t+1])
        end
        while length(findall(WEIGHT[:,t+1].>0))<Int(Threshold[1]*N)
            error_deduction[t+1] = CoolingSchedule*error_deduction[t+1]
            EPSILON[t+1] = EPSILON[t] - error_deduction[t+1]
            for i = 1:N
                WEIGHT[i,t+1] = WEIGHT[i,t] * (RW_Dist(XI[:,ANCESTOR[i,t],t])<EPSILON[t+1])
            end
        end
        WEIGHT[:,t+1] = WEIGHT[:,t+1]/sum(WEIGHT[:,t+1])
        if length(findall(WEIGHT[:,t+1].>0))<Int(Threshold[2]*N)
            ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
            WEIGHT[:,t+1] .= 1/N
        end
        Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2)
        accepted = zeros(N)
        proposed = zeros(N)
        Threads.@threads for i = 1:N
            XI[:,i,t+1],acc = RW_SMC_ABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t],K=K)
            accepted[i] = acc
            proposed[i] = K
            DISTANCE[i,t+1] = RW_Dist(XI[:,i,t+1])
        end
        acceptance = sum(accepted)/sum(proposed)
        SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(acceptance-0.234))
        #SIGMA[t+1] = SIGMA[t]
        ACCEPTANCE[t] = acceptance
    end
    (XI=XI,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,SIGMA=SIGMA,ACCEPTANCE=ACCEPTANCE,ANCESTOR=ANCESTOR)
end
=#
R2 = RW_SMC_ABC(10000,1500,20,Threshold=0.99,σ=0.3,λ=1.0,K=1)

using Plots, StatsPlots
theme(:mute)

plot(log.(R.EPSILON))
t = 1000
density(R.XI[4,findall(R.WEIGHT[:,t].> 0),t])



proposal_unique = zeros(2000)
ancestor_unique = zeros(2000)
current_unique  = zeros(2000)

for i = 1:2000
    proposal_unique[i] = length(unique(R.DISTANCE[R.ANCESTOR[:,i],i]))
    ancestor_unique[i] = length(unique(R.ANCESTOR[:,i]))
    current_unique[i]  = length(unique(R.DISTANCE[:,i+1]))
end
plot(proposal_unique); plot!(ancestor_unique)

plot(proposal_unique);plot!(current_unique)

plot(proposal_unique ./ ancestor_unique)

plot(R.ACCEPTANCE,ylim=(0,1))

plot(R.SIGMA)

quantile(unique(R.DISTANCE[R.ANCESTOR[:,1],1]),0.95)