using Distributions,LinearAlgebra,ProgressMeter
using ForwardDiff: gradient
f(u) = 3.0 + 1.0*(1+0.8*(1-exp(-2.0*u[2]))/(1+exp(-2.0*u[2])))*((1+u[2]^2)^u[1])*u[2];
function dist(u)
    return sum((f(u) .- ystar).^2)
end
C(x;ϵ) = ϵ - dist(x)
function logpi(u;ϵ)
    if C(u,ϵ=ϵ) > 0
        logpdf_k = sum(logpdf.(Uniform(0,10),u[1:2]))
        logpdf_z = logpdf(Normal(0,1),u[2])
        return logpdf_k + logpdf_z
    else
        return -Inf
    end
end

gradu(x) = gradient(x->sum(logpdf.(Uniform(0,1),x[1:2]))+logpdf(Normal(0,1),x[3]),x)
α_pu(x,u,δ;ϵ) = min(0,logpi(x.+δ*u,ϵ=ϵ)-logpi(x,ϵ=ϵ))

function BoundaryReflection(x0,u0,x1,u1,δ;ϵ)
    if any([x1[1]<0,x1[1]>10])
        # When the proposal steps outside the boundaries set by the prior
        u2   = reflect(u1,[sign(10-x1[1]),0])
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
function BPS(N,x0,u0,δ,refresh_rate;ϵ)
    X       = zeros(N,2)
    X[1,:]  = x0
    acc = 0
    @showprogress 1 "Computing.." for n = 2:N
        x1 = X[n-1,:] .+ δ*u0
        u1 = u0
        if any([x1[1]<0,x1[1]>10,C(x1,ϵ=ϵ)<=0])
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
function init(ϵ;NoData)
    x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
    while dist(x0) > ϵ
        x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
    end
    u0 = normalize(rand(Normal(0,1),4+NoData))
    return (x0,u0)
end
function scatterplot(X;color="red")
    scatter(X[:,1],X[:,2],color=color,label="",markersize=0.1,markerstrokewidth=0)
end

function ABC_BPS_Update(N,x0,u0,δ,refresh_rate;ϵ)
    X       = zeros(N,2)
    X[1,:]  = x0
    acc = 0
    for n = 2:N
        x1 = X[n-1,:] .+ δ*u0
        u1 = u0
        if any([x1[1]<0,x1[1]>10,C(x1,ϵ=ϵ)<=0])
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

function BPS_SMC_ABC(N,T;Threshold,δ,refresh_rate,K)
    U = zeros(2,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10)),rand(Normal(0,1))]
        DISTANCE[i,1] = dist(U[:,i,1])
    end
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    @showprogress 1 "Computing.." for t = 1:T
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Threads.@threads for i = 1:N
            if dist(U[:,ANCESTOR[i,t],t]) < EPSILON[t+1]
                U[:,i,t+1] = ABC_BPS_Update(K,U[:,ANCESTOR[i,t],t],normalize(rand(Normal(0,1),2)),δ,refresh_rate,ϵ=EPSILON[t+1])
                DISTANCE[i,t+1] = dist(U[:,i,t+1])
            end
        end
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR)
end

function RWMH(N,x0;ϵ,Σ,δ)
    X = zeros(N,2)
    X[1,:] = x0
    for n = 2:N
        xcand = X[n-1,:] .+ δ*rand(MultivariateNormal([0,0],Σ))
        α = min(0,logpi(xcand,ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
        else
            X[n,:] = X[n-1,:]
        end
    end
    return X[end,:]
end

function RW_SMC_ABC(N,T;Threshold,δ,K)
    U = zeros(2,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10)),rand(Normal(0,1))]
        DISTANCE[i,1] = dist(U[:,i,1])
    end
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    @showprogress 1 "Computing.."  for t = 1:T
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        Threads.@threads for i = 1:N
            if dist(U[:,ANCESTOR[i,t],t]) < EPSILON[t+1]
                U[:,i,t+1] = RWMH(K,U[:,ANCESTOR[i,t],t],ϵ=EPSILON[t+1],Σ=Σ,δ=δ)
                DISTANCE[i,t+1] = dist(U[:,i,t+1])
            end
        end
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR)
end
