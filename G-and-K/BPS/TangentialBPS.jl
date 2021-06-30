using Plots: position
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
    X = zeros(N+1,d)
    X[1,:] = x0
    AcceptNum = 0
    Proposals = zeros(N,d)
    PR = zeros(N)
    VR = zeros(N)
    CD = zeros(N)
    PD = zeros(N)
    U = zeros(N,d)
    ANGLES = zeros(N)
    @showprogress 1 "Computing.." for n = 2:(N+1)
        u0 = normalize(rand(Normal(0,1),d))
        U[n-1,:] = u0
        xcand,u2,ANGLES[n-1] = ϕ(X[n-1,:],u0;C=C,δ=δ,η=η)
        Proposals[n-1,:] = xcand
        CD[n-1] = C(X[n-1,:])
        PD[n-1] = C(xcand)
        position_ratio = logπ(xcand,ϵ=ϵ,y=y) - logπ(X[n-1,:],ϵ=ϵ,y=y)
        velocity_ratio = sum(logpdf.(Normal(0,1),u2)) - sum(logpdf.(Normal(0,1),u0))
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
    return (X=X,alpha=AcceptNum/N,PR=PR,VR=VR,A=ANGLES,CD=CD,PD=PD,Proposals=Proposals,U=U)
end



Random.seed!(123)
z0 = rand(Normal(0,1),20)
θ0 = [3.0,1.0,2.0,0.5]
dat20 = f.(z0,θ=θ0)
x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
while Euclidean(x0,y=dat20) >= 4.0
    x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
end




function f1(u;n,η)
    return normalize(u .- η*dot(u,n)*n)
end
function invf(v;n,η)
    normal_length = dot(v,n)
    tang_length   = norm(v .- dot(v,n)*n)
    uratio = normal_length/tang_length/(1-η)
    u_tang_length = sqrt(1/(1+uratio^2))
    u_normal_length = sqrt(1-u_tang_length^2)*sign(normal_length)
    tang_vec      = normalize(v .- dot(v,n)*n)
    return u_normal_length*n .+ u_tang_length*tang_vec
end
function ϕ2(x,u;C,δ,η)
    n1 = normalize(gradient(C,x))
    v1 = f1(u,n=n1,η=η)
    x2 = x .+ δ*v1
    n2 = normalize(gradient(C,x2))
    u2 = invf(-v1,n=n2,η=η)
    return (x2,u2)
end
function RW2(N,x0,δ;y,ϵ,η)
    d = length(x0)
    C(x) = Euclidean(x,y=y)
    X = zeros(N+1,d)
    X[1,:] = x0
    AcceptedNum = 0
    ProposeDist = zeros(N)
    CurrentDist = zeros(N)
    PR = zeros(N)
    @showprogress 1 "Computing..." for n = 2:(N+1)
        u0 = normalize(rand(Normal(0,1),d))
        xcand,_ = ϕ2(X[n-1,:],u0,C=C,δ=δ,η=η)
        CurrentDist[n-1] = C(X[n-1,:])
        ProposeDist[n-1] = C(xcand)
        position_ratio = logπ(xcand,ϵ=ϵ,y=y) - logπ(X[n-1,:],ϵ=ϵ,y=y)
        PR[n-1] = position_ratio
        alpha = min(0,position_ratio)
        if log(rand(Uniform(0,1))) < alpha
            X[n,:] = xcand
            AcceptedNum += 1
        else
            X[n,:] = X[n-1,:]
        end
    end
    return (X=X,α=AcceptedNum/N,PR =PR,CD = CurrentDist,PD = ProposeDist)
end

