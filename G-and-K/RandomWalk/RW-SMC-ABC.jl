# To make fair comparison, we set 
#=
function dist(ξ)
    return sum((f.(ξ[5:end],θ=ξ[1:4]) .- ystar).^2)
end
=#
using LinearAlgebra, Distributions
function dist(ξ)
    return norm(sort(f.(ξ[5:end],θ=ξ[1:4])) .- sort(ystar))
end

C(ξ;ϵ) = dist(ξ) - ϵ

function logpi(ξ;ϵ)
    if C(ξ,ϵ=ϵ) < 0
        logpdf_θ = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logpdf_z = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logpdf_θ + logpdf_z
    else
        return -Inf
    end
end

function RWMH(N,x0,ϵ,Σ,δ)
    D = length(x0)
    X = zeros(N,length(x0))
    X[1,:] = x0
    AcceptedNum = 0
    for n = 2:N
        #xcand = X[n-1,:] .+ δ*L*rand(Normal(0,1),D)
        xcand = rand(MultivariateNormal(X[n-1,:],δ^2*Σ))
        α = min(0,logpi(xcand,ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
            AcceptedNum += 1
        else
            X[n,:] = X[n-1,:]
        end
    end
    return (X[end,:],AcceptedNum/(N-1))
end
function RW_SMC_ABC(N,T,NoData;Threshold,δ,K)
    U = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = dist(U[:,i,1])
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
        println("SMC Step: ", t, " epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        # L = cholesky(Σ).L
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1],ParticleAcceptProb[index[i]] = RWMH(K,U[:,ANCESTOR[index[i],t],t],EPSILON[t+1],Σ,δ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = dist(U[:,index[i],t+1])
        end
        MH_AcceptProb[t] = mean(ParticleAcceptProb[index])
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,MH_AcceptProb)
end