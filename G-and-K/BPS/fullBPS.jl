using Distributions,LinearAlgebra,ProgressMeter, Random
using ForwardDiff: gradient
using Plots, StatsPlots
theme(:juno)
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function dist(ξ)
    return sum((f.(ξ[5:end],θ=ξ[1:4]) .- ystar).^2)
end

C(ξ;ϵ) = ϵ - dist(ξ)
function logpi(ξ;ϵ)
    if C(ξ,ϵ=ϵ) > 0
        logpdf_θ = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logpdf_z = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logpdf_θ + logpdf_z
    else
        return -Inf
    end
end
gradu(ξ) = gradient(x -> sum(logpdf.(Uniform(0,1),x[1:4]))+sum(logpdf.(Normal(0,1),x[5:end])),ξ)

Random.seed!(123)
zstar = rand(Normal(0,1),20)
θstar = [3,1,2,0.5]
ystar = f.(zstar,θ = θstar)

α_pu(ξ,u,δ;ϵ) = min(0,logpi(ξ .+ δ*u,ϵ=ϵ)-logpi(ξ,ϵ=ϵ))


function BoundaryReflection(x0,u0,x1,u1,δ;ϵ)
    if any([x1[1]<0,x1[1]>10,x1[2]<0,x1[2]>10,x1[3]<0,x1[3]>0,x1[4]<0,x1[4]>10])
        # When the proposal steps outside the boundaries set by the prior
        u2   = reflect(u1,[[sign(10-x1[1]),sign(10-x1[2]),sign(10-x1[3]),sign(10-x1[4])];repeat([0],length(u0)-4)])
        x2   = x1 .+ δ*u2
        α_dr = min(0,logpi(x2,ϵ=ϵ)-logpi(x0,ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α_dr
            return (x2,u2)
        else
            return (x0,-u0)
        end
    elseif C(x1,ϵ=ϵ) <= 0.0
        u2 = reflect(u1,gradient(x->C(x,ϵ=ϵ),x1))
        x2 = x1 .+ δ*u2
        α_dr = min(0,logpi(x2,ϵ=ϵ)-logpi(x0,ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α_dr
            return (x2,u2)
        else
            return (x0,-u0)
        end
    end
end
function EnergyReflection(x0,u0,x1,u1,δ;ϵ)
    u2 = reflect(u1,gradu(x1))
    x2 = x1 .+ δ*u2
    α_dr = min(0,log(1-exp(α_pu(x2,-u2,δ,ϵ=ϵ)))-log(1-exp(α_pu(x0,u0,δ,ϵ=ϵ)))+logpi(x2,ϵ=ϵ)-logpi(x0,ϵ=ϵ))
    if log(rand(Uniform(0,1))) < α_dr
        return (x2,u2)
    else
        return (x0,-u0)
    end
end
function DirectionRefresh(u0,refresh_rate)
    ind = rand(Bernoulli(refresh_rate))
    if ind == 0
        return u0
    else
        return normalize(rand(Normal(0,1),length(u0)))
    end
end
function reflect(u,v)
    v = normalize(v)
    return u - 2 * (transpose(u)*v)*v
end
function init(ϵ;NoData)
    x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
    while dist(x0) > ϵ
        x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
    end
    u0 = normalize(rand(Normal(0,1),4+NoData))
    return (x0,u0)
end
function BPS(N,x0,u0,δ,refresh_rate;ϵ,NoData)
    X       = zeros(N,4+NoData)
    X[1,:]  = x0
    acc = 0
    for n = 2:N
        x1 = X[n-1,:] .+ δ*u0
        u1 = u0
        if any([x1[1]<0,x1[1]>10,x1[2]<0,x1[2]>10,x1[3]<0,x1[3]>0,x1[4]<0,x1[4]>10,C(x1,ϵ=ϵ) <= 0.0])
            xhat,uhat = BoundaryReflection(X[n-1,:],u0,x1,u1,δ,ϵ=ϵ)
        else
            α = α_pu(X[n-1,:],u0,δ,ϵ=ϵ)
            if log(rand(Uniform(0,1))) < α
                xhat = x1
                uhat = u1
            else
                xhat,uhat = EnergyReflection(X[n-1,:],u0,x1,u1,δ,ϵ=ϵ)
            end
        end
        u0 = DirectionRefresh(uhat,refresh_rate)
        X[n,:] = xhat
        if norm(X[n-1,:] .- X[n,:]) > 1e-10
            acc += 1
        end 
    end
    return (X,acc/N)
end
function ABC_BPS_Update(N,x0,u0,δ,refresh_rate;ϵ,NoData)
    X       = zeros(N,4+NoData)
    X[1,:]  = x0
    acc = 0
    for n = 2:N
        x1 = X[n-1,:] .+ δ*u0
        u1 = u0
        if any([x1[1]<0,x1[1]>10,x1[2]<0,x1[2]>10,x1[3]<0,x1[3]>0,x1[4]<0,x1[4]>10,C(x1,ϵ=ϵ) <= 0.0])
            xhat,uhat = BoundaryReflection(X[n-1,:],u0,x1,u1,δ,ϵ=ϵ)
        else
            α = α_pu(X[n-1,:],u0,δ,ϵ=ϵ)
            if log(rand(Uniform(0,1))) < α
                xhat = x1
                uhat = u1
            else
                xhat,uhat = EnergyReflection(X[n-1,:],u0,x1,u1,δ,ϵ=ϵ)
            end
        end
        u0 = DirectionRefresh(uhat,refresh_rate)
        X[n,:] = xhat
        if norm(X[n-1,:] .- X[n,:]) > 1e-10
            acc += 1
        end 
    end
    return X[end,:]
end
function BPS_SMC_ABC(N,T,NoData;Threshold,δ,refresh_rate,K)
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
    for t = 1:T
        println("Iteration ",t)
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        @time Threads.@threads for i = 1:N
            if dist(U[:,ANCESTOR[i,t],t]) < EPSILON[t+1]
                U[:,i,t+1] = ABC_BPS_Update(K,U[:,ANCESTOR[i,t],t],normalize(rand(Normal(0,1),4+NoData)),δ,refresh_rate,ϵ=EPSILON[t+1],NoData=NoData)
                DISTANCE[i,t+1] = dist(U[:,i,t+1])
            end
        end
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR)
end
function RWMH(N,x0,ϵ,Σ,δ)
    D = length(x0)
    X = zeros(N,length(x0))
    X[1,:] = x0
    for n = 2:N
        xcand = rand(MultivariateNormal(X[n-1,:],δ^2*Σ))
        α = min(0,logpi(xcand,ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
        else
            X[n,:] = X[n-1,:]
        end
    end
    return X[end,:]
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
    for t = 1:T
    #@showprogress 1 "Computing.."  for t = 1:T
        println("Iteration ",t)
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Starting the parallel loop")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1] = RWMH(K,U[:,ANCESTOR[index[i],t],t],EPSILON[t+1],Σ,δ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = dist(U[:,index[i],t+1])
        end
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR)
end
