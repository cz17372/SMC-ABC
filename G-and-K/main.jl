using LinearAlgebra
using Distributions, Random, ProgressMeter, LinearAlgebra, Plots, StatsPlots

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
Euclidean(x;y) = norm(x .- y)
function Simulate(N;θ,d)
    Output = zeros(N,d)
    for n = 1:N
        Output[n,:] = f.(rand(Normal(0,1),d),θ=θ)
    end
    return Output
end

function SimulateOne(θ,d)
    return f.(rand(Normal(0,1),d),θ=θ)
end

function ABC_pseudoMCMC(N,θ0,x0;y,ϵ,δ,Σ)
    C(x) = Euclidean(x,y=y)
    M,d = size(x0)
    Output = zeros(N+1,length(θ0))
    Output[1,:] = θ0
    oldx = x0;
    Ind = 0
    @showprogress 1 "Computing..." for n = 2:(N+1)
        newθ = rand(MultivariateNormal(Output[n-1,:],δ^2*Σ))
        newx = Simulate(M,θ=newθ,d=d)
        if any(newθ .< 0.0) | any(newθ .> 10.0)
            Output[n,:] = Output[n-1,:]
        else
            oldkernel = sum(mapslices(C,oldx,dims=2)[:,1] .< ϵ)
            newkernel = sum(mapslices(C,newx,dims=2)[:,1] .< ϵ)
            if rand(Uniform(0,1)) < newkernel/oldkernel
                oldx = newx
                Output[n,:] = newθ
                Ind += 1
            else
                Output[n,:] = Output[n-1,:]
            end
        end
    end
    return (θ = Output, X = oldx, Prob = Ind/N)
end

function ABC_MCMC(N,θ0,x0;y,ϵ,δ,Σ)
    d = length(x0)
    Output = zeros(N+1,length(θ0))
    Output[1,:] = θ0
    oldx = x0;
    Ind = 0
    for n = 2:(N+1)
        newθ = rand(MultivariateNormal(Output[n-1,:],δ^2*Σ))
        if any(newθ .< 0.0) | any(newθ .> 10.0)
            Output[n,:] = Output[n-1,:]
        else
            newx = SimulateOne(newθ,d)
            if norm(newx .- y) < ϵ
                oldx = newx
                Output[n,:] = newθ
                Ind += 1
            else
                Output[n,:] = Output[n-1,:]
            end
        end
    end
    return (Output[end,:], oldx, Ind)
end
using Random
Random.seed!(123);
z0 = rand(Normal(0,1),20);
θ0 = [3.0,1.0,2.0,0.5];
dat20 = f.(z0,θ=θ0);
C(x) = Euclidean(x,y=dat20)
θ0 = rand(Uniform(0,10),4)
x0 = SimulateOne(θ0,20)
while C(x0) > 5.0
    θ0 = rand(Uniform(0,10),4)
    x0 = SimulateOne(θ0,20)
end


function SMC(N,T,y;Threshold,δ,K0,MinProb,MinStep)
    NoData = length(y)
    U = zeros(4,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    K = zeros(Int64,T+1)
    X = zeros(NoData,N,T+1)
    K[1] = K0
    for i = 1:N
        U[:,i,1] = rand(Uniform(0,10),4)
        X[:,i,1] = SimulateOne(U[:,i,1],NoData)
        DISTANCE[i,1] = norm(X[:,i,1] .- y)
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
        Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1],X[:,index[i],t+1],ParticleAcceptProb[index[i]] = ABC_MCMC(K[t],U[:,ANCESTOR[index[i],t],t],X[:,ANCESTOR[index[i],t],t],y=y,ϵ=EPSILON[t+1],δ=δ,Σ=Σ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = norm(X[:,index[i],t+1] .- y)
        end
        MH_AcceptProb[t] = mean(ParticleAcceptProb[index])/K[t]
        K[t+1] = Int64(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        if (MH_AcceptProb[t] < MinProb) & (δ > MinStep)
            δ = exp(log(δ) + 0.3*(MH_AcceptProb[t] - MinProb))
        end
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        println("The step size used in the next SMC iteration is ",δ)
        print("\n\n")
    end
    return (U=U,X=X,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K)
end

R = SMC(1000,50,dat20,Threshold=0.8,δ=0.3,K0=5,MinProb=0.25,MinStep=0.05)

index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[:,index,end]
density(X[1,:])


function SMC2(N,y;Threshold,δ,K0,MinProb,MinStep,StopProb,ϵT=0.5,Scheme="Adaptive")
    NoData = length(y)
    U = Array{Array{Float64,2},1}(undef,0)
    push!(U,zeros(4,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int,N,0)
    K = zeros(Int64,1); K[1] = K0
    X = Array{Array{Float64,2},1}(undef,0)
    push!(X,zeros(NoData,N))
    for i = 1:N
        U[1][:,i] = rand(Uniform(0,10),4)
        X[1][:,i] = SimulateOne(U[1][:,i],NoData)
        DISTANCE[i,1] = norm(X[1][:,i] .- y)
    end
    WEIGHT[:,1] .= 1.0/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    ParticleAcceptProb = zeros(N)
    MH_AcceptProb = zeros(1); MH_AcceptProb[end] = 1.0
    t = 0
    while (MH_AcceptProb[end] > StopProb) & (EPSILON[end]>=ϵT)
        t += 1
        ANCESTOR = hcat(ANCESTOR,vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...));
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            push!(EPSILON,quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold))
        else
            push!(EPSILON, findmax(unique(DISTANCE[ANCESTOR[:,t],t]))[1])
        end
        WEIGHT = hcat(WEIGHT,(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1]))
        println("SMC Step: ", t)
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("K = ", K[t])
        Σ = cov(U[t][:,findall(WEIGHT[:,t].>0)],dims=2) + 1e-8*I
        index = findall(WEIGHT[:,t+1] .> 0.0)
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        push!(U,zeros(4,N)); push!(X,zeros(NoData,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        @timed Threads.@threads for i = 1:length(index)
            U[t+1][:,index[i]],X[t+1][:,index[i]],ParticleAcceptProb[index[i]] = ABC_MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],X[t][:,ANCESTOR[index[i],t]],y=y,ϵ=EPSILON[t+1],δ=δ,Σ=Σ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = norm(X[t+1][:,index[i]] .- y)
        end
        push!(MH_AcceptProb,mean(ParticleAcceptProb[index])/K[end])
        if Scheme=="Adaptive"
            push!(K,Int64(ceil(log(0.01)/log(1-MH_AcceptProb[end]))))
        elseif Scheme == "Fixed"
            push!(K,K0)
        end
        if (MH_AcceptProb[end] < MinProb) & (δ > MinStep)
            δ = exp(log(δ) + 0.5*(MH_AcceptProb[end] - MinProb))
        end
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        println("The step size used in the next SMC iteration is ",δ)
        print("\n\n")
    end
    return (U=U,X=X,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb=MH_AcceptProb,K=K)
end


R = SMC2(5000,dat20,Threshold=0.99,δ = 0.1, K0 = 20, MinProb = 0.1, MinStep = 0.1, StopProb = 0.01,Scheme="Fixed")
plot!(log.(R.EPSILON),label="")

indices = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,indices]
density(X[4,:])