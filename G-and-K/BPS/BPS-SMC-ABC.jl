module BPS
using LinearAlgebra, Distributions
using ForwardDiff: gradient

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

# Defines the boundary for constrained region, parameterized by ϵ
Dist(x;y) = norm(f.(x[5:end],θ=x[1:4]) .- y)
C(x;y,ϵ)  = Dist(x,y=y) - ϵ

logPrior(x) = sum(logpdf.(Uniform(0,10),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
# define the log-pdf of the prior, constrained by C(ξ;y,ϵ)
function logpi(x::Vector{Float64},u::Vector{Float64};y::Vector{Float64},ϵ::Float64,Σ)
    if C(x,y=y,ϵ=ϵ) > 0
        return -Inf
    else
        return logPrior(x) + logpdf(MultivariateNormal(zeros(length(u)),Σ),u)
    end
end

# Define the energy function for internal reflection 
U(x) = -logPrior(x)


#---------------------------- Proposal Mechanism ------------------------------#
export φ1, EnergyBounce, BoundaryBounce, σ, φ2
function φ1(x0::Vector{Float64},u0::Vector{Float64},δ::Float64)
    return (x0 .+ δ*u0, -u0)
end

function EnergyBounce(x0,u0;gradFunc)
    n = normalize(gradient(gradFunc,x0))
    return u0 .- 2.0*dot(u0,n)*n
end

function BoundaryBounce(x0,u0;gradFunc)
    if any([(x0[1:4].<0);(x0[1:4].>10)])
        n = normalize(1.0*(x0 .< 0) .- 1.0*(x0 .> 10))
    else
        n = normalize(gradient(gradFunc,x0))
    end
    return u0 .- 2.0*dot(u0,n)*n
end

σ(x0,u0) = (x0,-u0)

function φ2(x0,u0,δ;BounceType,gradFunc)
    x1, u1 = φ1(x0,u0,δ)
    xflip,uflip = σ(x1,u1)
    xbounce = xflip; ubounce = BounceType(xflip,uflip,gradFunc=gradFunc)
    x2,u2 = φ1(xbounce,ubounce,δ)
    return x2,u2
end

#----------------------------- Acceptance Probability ----------------------------#
function α1(x0,u0,δ;y,ϵ,Σ)
    x1,u1 = φ1(x0,u0,δ)
    return min(0,logpi(x1,u1,y=y,ϵ=ϵ,Σ=Σ) - logpi(x0,u0,y=y,ϵ=ϵ,Σ=Σ))
end

function α2(x0,u0,δ;y,ϵ,Σ,BounceType,gradFunc)
    x2,u2 = φ2(x0,u0,δ,BounceType=BounceType,gradFunc=gradFunc)
    forward_1st_proposal_acc = exp(α1(x0,u0,δ,y=y,ϵ=ϵ,Σ=Σ))
    backward_1st_proposal_acc = exp(α1(x2,u2,δ,y=y,ϵ=ϵ,Σ=Σ))
    return min(0,log(1-backward_1st_proposal_acc)-log(1-forward_1st_proposal_acc)+logpi(x2,u2,y=y,ϵ=ϵ,Σ=Σ)-logpi(x0,u0,y=y,ϵ=ϵ,Σ=Σ))
end

#--------------------------- Veclocity Update ------------------------------------#
function DirectionRefresh(u0,κ,Σ)
    ind = rand(Bernoulli(κ))
    if ind == 0
        return u0
    else
        return rand(MultivariateNormal(zeros(length(u0)),Σ))
    end
end

#--------------------------- BPS Sampler ------------------------------------------#
function BPS_LocalMH(N,x0,δ,κ;y,ϵ,Σ)
    X = zeros(N,length(x0))
    X[1,:] = x0
    u0 = rand(MultivariateNormal(zeros(length(x0)),Σ))
    AcceptedNum = 0
    boundfunc(x) = C(x,y=y,ϵ=ϵ)
    for n = 2:N
        x1,u1 = φ1(X[n-1,:],u0,δ)
        if (any([(x1[1:4].>10);(x1[1:4] .< 0)])) || (Dist(x1,y=y) >= ϵ)
            x2,u2 = φ2(X[n-1,:],u0,δ,BounceType=BoundaryBounce,gradFunc=boundfunc)
            alpha2 = α2(X[n-1,:],u0,δ;y=y,ϵ=ϵ,Σ=Σ,BounceType=BoundaryBounce,gradFunc=boundfunc)
            if log(rand(Uniform(0,1))) < alpha2
                AcceptedNum += 1
                xhat=x2
                uhat=u2
            else
                xhat=X[n-1,:]
                uhat=u0
            end
        else
            alpha1 = α1(X[n-1,:],u0,δ;y=y,ϵ=ϵ,Σ=Σ)
            if log(rand(Uniform(0,1))) < alpha1
                AcceptedNum += 1
                xhat = x1
                uhat = u1
            else
                x2,u2 = φ2(X[n-1,:],u0,δ,BounceType=EnergyBounce,gradFunc=U)
                alpha2 = α2(X[n-1,:],u0,δ;y=y,ϵ=ϵ,Σ=Σ,BounceType=EnergyBounce,gradFunc=U)
                if log(rand(Uniform(0,1))) < alpha2
                    xhat = x2
                    uhat = u2
                    AcceptedNum += 1
                else
                    xhat = X[n-1,:]
                    uhat = u0
                end
            end
        end
        # Velocity flip
        xhat,uhat = σ(xhat,uhat)
        # Refresh Velocity
        X[n,:] = xhat
        u0 = DirectionRefresh(uhat,κ,Σ)
    end
    return X[end,:],AcceptedNum
end

#-------------------------- BPS_SMC_ABC -------------------------------------------#
function BPS_SMC_ABC(N,T,y;Threshold,δ,κ,K0)
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
        index = findall(WEIGHT[:,t+1] .> 0.0)          
        Σ = cov(U[:,index,t],dims=2) + 1e-8*I
        d = mean(mapslices(norm,rand(MultivariateNormal(zeros(4+NoData),Σ),100000),dims=1))
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1],ParticleAcceptProb[index[i]] = BPS_LocalMH(K[t],U[:,ANCESTOR[index[i],t],t],δ,κ,y=y,ϵ=EPSILON[t+1],Σ = 1.0/d^2*Σ)
            DISTANCE[index[i],t+1] = Dist(U[:,index[i],t+1],y=y)
        end
        MH_AcceptProb[t] = mean(ParticleAcceptProb[index])/K[t]
        K[t+1] = Int64(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        print("\n\n")
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K)
end
end

