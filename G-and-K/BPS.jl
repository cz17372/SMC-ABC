using Distributions, Plots, StatsPlots, LinearAlgebra, Random
using ForwardDiff: gradient

# function used to transform standard Normal to g-and-k
f(u) = 3.0 + 1.0*(1+0.8*(1-exp(-2.0*u[2]))/(1+exp(-2.0*u[2])))*((1+u[2]^2)^u[1])*u[2];
Random.seed!(12358)
zstar = rand(Normal(0,1))
ystar = f([0.5,zstar])

function dist(u)
    return sum((f(u) .- ystar).^2)
end
C(x;ϵ=ϵ) = ϵ - dist(x)
function logpi(u;ϵ)
    if C(u,ϵ=ϵ) > 0
        logpdf_k = logpdf(Uniform(0,10),u[1])
        logpdf_z = logpdf(Normal(0,1),u[2])
        return logpdf_k + logpdf_z
    else
        return -Inf
    end
end

gradu(x) = gradient(x->logpdf(Uniform(0,1),x[1])+logpdf(Normal(0,1),x[2]),x)
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
        return normalize(rand(Normal(0,1),2))
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
    for n = 2:N
        x1 = X[n-1,:] .+ δ*u0
        u1 = u0
        if any([x1[1]<0,x1[1]>10,C(x1)<=0])
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

ϵ=5
x0 = [rand(Uniform(0,10)),rand(Normal(0,1))]
while dist(x0) > ϵ
    x0 = [rand(Uniform(0,10)),rand(Normal(0,1))]
end
u0 = normalize(rand(Normal(0,1),2))

# Sample Ground truth
truth = zeros(10000,2)
n = 1
while n <= 10000
    ucand = [rand(Uniform(0,10)),rand(Normal(0,1))]
    if C(ucand) > 0
        truth[n,:] = ucand
        n += 1
    end
end
X,acc = BPS(10000,x0,u0,0.1,0.5,ϵ=ϵ)
scatter(truth[:,1],truth[:,2],markersize=0.1,markerstrokewidth=0,color=:red,label="")
scatter!(X[:,1],X[:,2],markersize=0.1,markerstrokewidth=0,color=:white,label="")