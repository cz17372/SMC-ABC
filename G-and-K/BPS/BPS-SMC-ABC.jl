module BPS
using LinearAlgebra, Distributions
using ForwardDiff: gradient

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

# Defines the boundary for constrained region, parameterized by ϵ
#Dist(x;y) = norm(sort(f.(x[5:end],θ=x[1:4])) .- sort(y))
Dist(x;y) = norm(f.(x[5:end],θ=x[1:4]) .- y)
C(x;y,ϵ)  = Dist(x,y=y) - ϵ

logPrior(x) = sum(logpdf.(Uniform(0,10),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
# define the log-pdf of the prior, constrained by C(ξ;y,ϵ)

function logpi(x::Vector{Float64};y::Vector{Float64},ϵ::Float64)
    if C(x,y=y,ϵ=ϵ) > 0
        return -Inf
    else
        return logPrior(x)
    end
end

# Define the energy function for internal reflection 
U(x) = -logPrior(x)


#---------------------------- Proposal Mechanism ------------------------------#
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
function α1(x0,u0,δ;y,ϵ)
    x1,u1 = φ1(x0,u0,δ)
    return min(0,logpi(x1,y=y,ϵ=ϵ) - logpi(x0,y=y,ϵ=ϵ))
end

function α2(x0,u0,δ;y,ϵ,BounceType,gradFunc)
    x2,u2 = φ2(x0,u0,δ,BounceType=BounceType,gradFunc=gradFunc)
    forward_1st_proposal_acc = exp(α1(x0,u0,δ,y=y,ϵ=ϵ))
    backward_1st_proposal_acc = exp(α1(x2,u2,δ,y=y,ϵ=ϵ))
    return min(0,log(1-backward_1st_proposal_acc)-log(1-forward_1st_proposal_acc)+logpi(x2,y=y,ϵ=ϵ)-logpi(x0,y=y,ϵ=ϵ))
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
function BPS1(N::Int64,x0::Vector{Float64},δ::Float64,κ::Float64;y::Vector{Float64},ϵ::Float64)
    boundfunc(x) = C(x,y=y,ϵ=ϵ)
    X = zeros(N,length(x0))
    X[1,:] = x0
    u0 = normalize(rand(Normal(0,1),length(x0)))
    AcceptedNumber = 0; BoundaryBounceProposed = 0; BoundaryBounceAccepted = 0;
    for n = 2:N
        #println(n)
        x1,u1 = φ1(X[n-1,:],u0,δ)
        if (any([(x1[1:4].>10);(x1[1:4] .< 0)])) || (Dist(x1,y=y) >= ϵ)
            BoundaryBounceProposed += 1
            x2,u2 = φ2(X[n-1,:],u0,δ,BounceType=BoundaryBounce,gradFunc=boundfunc)
            iter = 1
            while (any([(x2[1:4].>10);(x2[1:4] .< 0)])) || (Dist(x2,y=y) >= ϵ)
                iter += 1
                x2,u2 = φ2(x2 .+ δ*u2,-u2,δ,BounceType=BoundaryBounce,gradFunc=boundfunc)
                if iter > 10
                    break
                end
            end
            alpha2 = logpi(x2,y=y,ϵ=ϵ) - logpi(X[n-1,:],y=y,ϵ=ϵ)
            if log(rand(Uniform(0,1))) < alpha2
                BoundaryBounceAccepted += 1
                AcceptedNumber += 1
                xhat=x2
                uhat=u2
            else
                xhat=X[n-1,:]
                uhat=u0
            end
        else
            alpha1 = α1(X[n-1,:],u0,δ;y=y,ϵ=ϵ)
            if log(rand(Uniform(0,1))) < alpha1
                AcceptedNumber += 1
                xhat = x1
                uhat = u1
            else
                x2,u2 = φ2(X[n-1,:],u0,δ,BounceType=EnergyBounce,gradFunc=U)
                alpha2 = α2(X[n-1,:],u0,δ;y=y,ϵ=ϵ,BounceType=EnergyBounce,gradFunc=U)
                if log(rand(Uniform(0,1))) < alpha2
                    AcceptedNumber += 1
                    xhat = x2
                    uhat = u2
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
        u0 = DirectionRefresh(uhat,δ,κ)
    end
    return X[end,:],BoundaryBounceProposed,BoundaryBounceAccepted,AcceptedNumber
end

"""
    BPS2(N,x0,δ,κ,y,ϵ,Σ)

Perform the second type of Discrete Bouncy Particle Sampler Algorithm. Boundary bounce takes place at the original position when the first proposal steps outside the boundary

# Arguments
- `N::Integer`: The number of iterations the discrete BPS algorithm performs
- `x0::Vector`: Starting position of the discrete BPS algorithm
- `δ::Float`  : Step size of the discrete BPS algorithm
- `κ::Float`  : Velocity refreshment rate for the velocity component
"""
function BPS2(N::Int64,x0::Vector{Float64},δ::Float64,κ::Float64;y::Vector{Float64},ϵ::Float64)
    boundfunction(x) = C(x,y=y,ϵ=ϵ) # Equation of the boundary for given "y" and "ϵ"
    X = zeros(N+1,length(x0))
    X[1,:] = x0
    # Random generate a starting velocity
    u0 = normalize(rand(Normal(0,1),length(x0)))
    AcceptedNumber = 0; BoundaryBounceProposed = 0; BoundaryBounceAccepted = 0;
    for n = 2:(N+1)
        # generate the first proposal according to ϕ1
        x1,u1 = φ1(X[n-1,:],u0,δ)
        # Check if x1 steps outside the boundary
        if any([(x1[1:4].>10) ; (x1[1:4] .< 0)]) || (boundfunction(x1)>0)
            # The boundary bounce is performed
            BoundaryBounceProposed += 1
            x2,u2 = X[n-1,:], -BoundaryBounce(X[n-1,:],u0,gradFunc = U)
            # The acceptance probability for this move is given by (1-α1∘ ϕ2(x0,u0))/(1-α1(x0,u0))
            alpha = log(1-exp(α1(x2,u2,δ,y=y,ϵ=ϵ)))-log(1-exp(α1(X[n-1,:],u0,δ,y=y,ϵ=ϵ)))
            if log(rand(Uniform(0,1))) < alpha
                BoundaryBounceAccepted += 1
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = u0
            end
        else
            # The first proposal is not outside the boundary
            alpha1 = α1(X[n-1,:],u0,δ,y=y,ϵ=ϵ)
            if log(rand(Uniform(0,1))) < alpha1
                # The first proposal is accepted 
                AcceptedNumber += 1
                xhat = x1
                uhat = u1
            else
                # Upon rejection of the first proposal, propose a second move by bounce operation
                x2,u2 = φ2(X[n-1,:],u0,δ,BounceType=EnergyBounce,gradFunc=U)
                alpha2 = α2(X[n-1,:],u0,δ,y=y,ϵ=ϵ,BounceType=EnergyBounce,gradFunc=U)
                if log(rand(Uniform(0,1))) < alpha2
                    # The second proposla is accepted
                    AcceptedNumber += 1
                    xhat = x2
                    uhat = u2
                else
                    xhat = X[n-1,:]
                    uhat = u0
                end
            end
        end
        # Flip the velocity
        xhat,uhat = σ(xhat,uhat)
        X[n,:] = xhat
        # Refresh the velocity
        u0 = DirectionRefresh(xhat,δ,κ)
    end
    return X[end,:],BoundaryBounceProposed,BoundaryBounceAccepted,AcceptedNumber
end


#-------------------------- BPS_SMC_ABC -------------------------------------------#
"""
    SMC(N,T,y;Threshold,δ,κ,K0,MH)
Perform SMC-ABC algorithm with Discrete Bouncy Particle Sampler algorithm for local exploration

# Arguments
- `N::Int` : The number of particles use for each SMC step
- `T::Int` : The number of SMC steps performed 
- `y::Vector`: The observations we have in hand
-  `Threshold::Float` : A float number between 0 and 1 representing the threshold used for determining the next value of ϵ. Upon resampling from the particles of the previous iterations, then the next ϵ is chosen such that `Threshold` × `Total number of unique particles after resampling` of the unique particles will be kept alive
- `δ::Float` : The stepsize for the discrete BPS algorithm
- `κ::Float` : The velocity refreshment rate for the discrete BPS algorithm
- `K0::Float`: The initial number of iterations for each discrete BPS chain
- `MH`       : The BPS sampler used within the SMC algorithm. Current choice includes `BPS1` and `BPS2`
"""
function SMC(N::Int64,T::Int64,y::Vector{Float64};Threshold::Float64,δ::Float64,κ::Float64,K0::Int64,MH)
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
    BoundaryBounceTimeVec = zeros(N); BoundaryBounceSuccessVec = zeros(N)
    BoundaryBounceAccepted = zeros(T); BoundaryBounceProposed = zeros(T); 
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
            U[:,index[i],t+1], BoundaryBounceTimeVec[index[i]], BoundaryBounceSuccessVec[index[i]], ParticleAccepted[index[i]] = MH(K[t],U[:,ANCESTOR[index[i],t],t],δ,κ,y=y,ϵ=EPSILON[t+1])
            DISTANCE[index[i],t+1] = Dist(U[:,index[i],t+1],y=y)
        end
        MH_AcceptProb[t] = mean(ParticleAccepted[index])/(K[t])
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        BoundaryBounceProposed[t] = sum(BoundaryBounceTimeVec[index])
        println("Proportion of internal proposal is ",1 - BoundaryBounceProposed[t]/(length(index)*K[t]))
        BoundaryBounceAccepted[t] = sum(BoundaryBounceSuccessVec[index])
        K[t+1] = Int(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        if MH_AcceptProb[t] < 0.5
            δ = exp(log(δ) + 0.3*(MH_AcceptProb[t] - 0.5))
        end
        println("The step size used in the next SMC iteration is ",δ)
        print("\n\n")
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K,BoundaryBounceAccepted=BoundaryBounceAccepted,BoundaryBounceProposed=BoundaryBounceProposed)
end
end

