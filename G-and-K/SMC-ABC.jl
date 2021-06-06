using LinearAlgebra, Distributions
using ForwardDiff: gradient
using ProgressMeter
using Plots, StatsPlots
using Random
using Roots
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
# Defines the boundary for constrained region, parameterized by ϵ
Dist1(x;y) = norm(sort(f.(x[5:end],θ=x[1:4])) .- sort(y))
Dist2(x;y) = norm(f.(x[5:end],θ=x[1:4]) .- y)
C(x;y,ϵ,Dist)  = Dist(x,y=y) - ϵ
object(k;x0,u0,C) = prod((x0 .+ k*u0)[1:4])*prod((x0 .+ k*u0)[1:4] .- 10.0)*C(x0 .+  k*u0)
prior_boundary(x0) = any([(abs.(x0[1:4]) .< 1e-15);(abs.(x0[1:4] .- 10.0) .< 1e-15)])
get_prior_normal(x0) = normalize([(abs.(x0[1:4]) .< 1e-15) .+ (abs.(x0[1:4] .- 10.0) .< 1e-15);zeros(length(x0[5:end]))])
# define the log-pdf of the prior, constrained by C(ξ;y,ϵ)
function logpi(x::Vector{Float64};y::Vector{Float64},ϵ::Float64,Dist)
    if C(x,y=y,ϵ=ϵ,Dist=Dist) > 0
        return -Inf
    else
        return logPrior(x)
    end
end
# Define the energy function for internal reflection 
U(x) = sum(logpdf.(Uniform(0,10),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
#---------------------------- Proposal Mechanism ------------------------------#
function φ1(x0,u0,δ;C)
    output = x0
    working_delta = δ
    intermediate_x = copy(x0)
    intermediate_u = copy(u0)
    roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    boundary_bounce = 0
    No_Bounces = 0
    while length(roots) > 0
        No_Bounces += 1
        k = roots[1]
        intermediate_x = intermediate_x .+ k*intermediate_u
        output = hcat(output,intermediate_x)
        working_delta -= k
        if prior_boundary(intermediate_x)
            n = get_prior_normal(intermediate_x)
        else
            boundary_bounce += 1
            n = normalize(gradient(C,intermediate_x))
        end
        intermediate_u = intermediate_u .- 2*dot(intermediate_u,n)*n
        roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    end
    output = hcat(output,intermediate_x .+ working_delta * intermediate_u)

    return output[:,end], -intermediate_u, No_Bounces
end

σ(x0,u0) = (x0,-u0)
#----------------------------- Acceptance Probability ----------------------------#
function α1(x1,x0)
    return min(0,U(x1) - U(x0))
end
function α2(x2,x1,x0)
    forward_rejection = log(1 - exp(α1(x1,x0)))
    backward_rejection = log(1 - exp(α1(x1,x2)))
    return min(0,backward_rejection+U(x2)-forward_rejection-U(x0))
end

#--------------------------- Veclocity Update ------------------------------------#
function DirectionRefresh(u0,δ,κ)
    p = exp(-κ*δ)
    ind = rand(Bernoulli(p))
    if ind == 1
        return u0
    else
        return normalize(rand(Normal(0,1),length(u0)))
    end
end

#--------------------------- BPS Sampler ------------------------------------------#
function BPS1(N::Int64,x0::Vector{Float64},δ::Float64,κ::Float64;y::Vector{Float64},ϵ::Float64,Dist)
    C0(x) = C(x,y=y,ϵ=ϵ,Dist=Dist)
    X = zeros(N+1,length(x0))
    X[1,:] = x0
    u0 = normalize(rand(Normal(0,1),length(x0)))
    AcceptedNumber = 0; Ind = 0; Bounces = 0
    for n = 2:(N+1)
        #println(n)
        x1,u1,b = φ1(X[n-1,:],u0,δ,C=C0)
        Ind += 1; Bounces += b
        #println("Number of bounces = ",b)
        if log(rand(Uniform(0,1))) < α1(x1,X[n-1,:])
            AcceptedNumber += 1
            xhat = x1
            uhat = u1
        else
            dir = normalize(gradient(U,x1)); 
            ubound = -u1 .- 2.0 * dot(-u1,dir) * dir
            x2,u2,b = φ1(x1,ubound,δ,C=C0)
            Ind += 1; Bounces += b
            #println("Number of bounces = ",b)
            if log(rand(Uniform(0,1))) < α2(x2,x1,X[n-1,:])
                AcceptedNumber += 1
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = u0
            end
        end
        xhat,uhat = σ(xhat,uhat)
        u0 = DirectionRefresh(uhat,δ,κ)
        X[n,:] = xhat
        #println(C0(X[n,:])+ϵ)
    end
    return X[end,:], AcceptedNumber, Bounces/Ind
end

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
# Set the true static parameters
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);

samp() = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
ϵ = 5.0
x0 = samp()
while Dist2(x0,y=dat20) >= ϵ
    x0 = samp()
end

R = BPS1(50000,x0,0.2,0.5,y=dat20,ϵ=5.0,Dist=Dist2)

function SMC(N::Int64,T::Int64,y::Vector{Float64};Threshold::Float64,δ::Float64,κ::Float64,K0::Int64,MH,Dist)
    NoData = length(y)
    U = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    K = zeros(Int64,T+1)
    K[1] = K0
    for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = Dist(U[:,i,1],y=y)
    end
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    ParticleAccepted = zeros(N)
    MH_AcceptProb = zeros(T)
    BounceNoVec = zeros(N)
    AveBounceNo = zeros(T)
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
        index = findall(WEIGHT[:,t+1] .> 0.0)          
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1], ParticleAccepted[index[i]],BounceNoVec[index[i]] = MH(K[t],U[:,ANCESTOR[index[i],t],t],δ,κ,y=y,ϵ=EPSILON[t+1],Dist=Dist)
            DISTANCE[index[i],t+1] = Dist(U[:,index[i],t+1],y=y)
        end
        MH_AcceptProb[t] = mean(ParticleAccepted[index])/(K[t])
        AveBounceNo[t]   = mean(BounceNoVec[index])
        println("Average Number of Bounces per proposal = ",AveBounceNo[t])
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        if MH_AcceptProb[t] >= 1.0
            K[t+1] = 1
        else
            K[t+1] = Int(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        end
        if MH_AcceptProb[t] < 0.5
            δ = exp(log(δ) + 0.3*(MH_AcceptProb[t] - 0.5))
        end
        println("The step size used in the next SMC iteration is ",δ)
        print("\n\n")
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K)
end

u0 = normalize(rand(Normal(0,1),24))
C0(x) = C(x,y=dat20,ϵ=ϵ,Dist=Dist2)
X,u = φ1(x0,u0,0.5,C = C0)
X2,u2 =  φ1(x0,u0,0.5,C = C0)
α2(X2,X,x0)
Dist2(X2,y=dat20)

plot(R[1][:,2])

density(R[1][:,1])

R = SMC(2000,300,dat20,Threshold=0.8,δ=0.3,κ=1.0,K0 = 5,MH=BPS1,Dist=Dist2)

index = findall(R.WEIGHT[:,end] .> 0)
density(R.U[3,index,end])

R2 = SMC(1000,200,dat20,Threshold=0.8,δ=0.5,κ=1.0,K0 = 5,MH=BPS1,Dist=Dist2)
index = findall(R2.WEIGHT[:,end] .> 0)
density(R2.U[1,index,end])

R3 = SMC(1000,200,dat20,Threshold=0.8,δ=0.4,κ=1.0,K0 = 5,MH=BPS1,Dist=Dist2)
index = findall(R3.WEIGHT[:,end] .> 0)
density!(R3.U[1,index,end])


R4 = SMC(2000,200,dat20,Threshold=0.8,δ=0.2,κ=2.0,K0 = 5,MH=BPS1,Dist=Dist2)
index = findall(R4.WEIGHT[:,end] .> 0)
density(R4.U[2,index,end])
plot(log.(R4.EPSILON))

x0 = R3.U[:,1,end]


X,Acc,Bounce = BPS1(20000,x0,0.05,2.0,y=dat20,ϵ=1.0,Dist=Dist2)

distances= (mapslices(x->Dist2(x,y=dat20),X,dims=2))


density(X[15001:end,4])

plot(X[:,4])