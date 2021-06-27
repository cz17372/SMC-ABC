using LinearAlgebra
using Distributions, Random
using ForwardDiff:gradient 
using LinearAlgebra
using Plots,StatsPlots
using ProgressMeter
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
g(x)   = f.(x[5:end],θ=x[1:4])
Euclidean(x;y) = norm(g(x) .- y)


"""
    logπ(x;ϵ,y)
The log-target density in the ABC context. 

# Arguments
- `x::Vector{Float}`: uniorn of the static parameters (first four elements) and the latents (the size of the latents should be the same as the observations)
- `ϵ::Float`: tolerance level under the Euclidean distance between the psuedo-observations and the actual observations
- ``
"""
function logπ(x;ϵ,y)
    if Euclidean(x,y=y) >= ϵ
        return -Inf
    else
        return sum(logpdf.(Uniform(0,10),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
    end
end


function ϕ(x,u;C,δ,η)
    n1 = normalize(gradient(C,x))
    A1 = I - η*n1*transpose(n1)
    x2 = x .+ δ*A1*u
    n2 = normalize(gradient(C,x2))
    A2 = I - η*n2*transpose(n2)
    u2 = - inv(A2)*A1*u
    θ = acosd(dot(n1,n2)) 
    return (x2,u2,θ)
end


function RW(N,x0,δ;y,ϵ,η)
    d = length(x0)
    C(x) = Euclidean(x,y=y)
    X = zeros(N,length(x0))
    X[1,:] = x0
    AcceptNum = 0
    PR = zeros(N-1)
    VR = zeros(N-1)
    ANGLES = zeros(N-1)
    @showprogress 1 "Computing.." for n = 2:N
        u0 = normalize(rand(Normal(0,1),d))
        xcand,u2,ANGLES[n-1] = ϕ(X[n-1,:],u0;C=C,δ=δ,η=η)
        position_ratio = logπ(xcand,ϵ=ϵ,y=y) - logπ(X[n-1,:],ϵ=ϵ,y=y)
        velocity_ratio = sum(logpdf.(Normal(0,1/sqrt(d)),u2)) - sum(logpdf.(Normal(0,1/sqrt(d)),u0))
        PR[n-1] = position_ratio
        VR[n-1] = velocity_ratio
        α = min(0,position_ratio+velocity_ratio)
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
            AcceptNum += 1
        else
            X[n,:] = X[n-1,:]
        end
    end
    return (X,AcceptNum/(N-1),PR,VR,ANGLES)
end

x0
u0 = rand(Normal(0,1/sqrt(24)),24)
C(x) = Euclidean(x,y=dat20)
x2,u2 = ϕ(x0,u0,C=C,δ=0.3,η=0.95)
x3,u3 = ϕ(x2,u2,C=C,δ=0.3,η=0.95)


Random.seed!(123)
z0 = rand(Normal(0,1),20)
θ0 = [3.0,1.0,2.0,0.5]
dat20 = f.(z0,θ=θ0)
x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
while Euclidean(x0,y=dat20) >= 4.0
    x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
end

R, alpha, PR, VR, Angles = RW(10000,U[:,1],0.1;y=dat20,ϵ=0.5,η=0.9)

mean(VR)

mean(PR[PR .!= -Inf])
mean(PR .== -Inf)
mean(Angles)

plot(R[:,1])

plot(Angles)
scatter(Angles,VR)

x = rand(Normal(0,1),10000,2)
scatter(x[:,1],x[:,2])

u = normalize(rand(10))
n = normalize(rand(10))
eta = 0.8
v = normalize(u .- eta*dot(u,n)*n)

k = dot(v,n)/norm(v .- dot(v,n)*n)/(1-eta)
lt = sqrt(1/(1+k^2))
ln = sqrt(1 - lt^2)
dot(u,n)/norm(u .- dot(u,n)*n)

ru = ln*n .+ lt*normalize(v .- dot(v,n)*n)

x0 = U[:,1]
u0 = normalize(rand(Normal(0,1),24))
x1 = ϕ(x0,u0,C = C,δ=0.1,η=0.99)[1]
logπ(x1,ϵ=0.5,y=dat20)
plot(R[:,1])

function f(x,u;C,η)
    n = normalize(gradient(C,x))
    return normalize(u .- η*dot(u,n)*n)
end


function invf(x,v;C,η)
    

