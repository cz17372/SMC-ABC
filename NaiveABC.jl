using Distributions, LinearAlgebra, Random,Optim
using ProgressMeter
using StatsPlots, Plots
f(z;θ) = θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
d(θ;z) = norm(f.(z,θ=θ) .- y0)
logPrior(θ) = sum(logpdf.(Uniform(0,10),θ))
function ESJD_NaiveABC(sigma;P0,U,Z,epsilon,dist)
    N,C = size(P0)
    NewP = P0 .+ sigma*U
    JumpD = (mapslices(norm,sigma*U,dims=2)[:,1]).^2
    α = zeros(N)
    for i = 1:N
        α[i] = exp(min(0,logPrior(NewP[i,:])-logPrior(P0[i,:])+log(dist(NewP[i,:],z=Z[i,:])<epsilon)))
    end
    return mean(JumpD .* α)
end

function MHUpdate_NaiveABC(sigma;P0,U,Z,epsilon,dist)
    N,C=size(P0)
    NewP = P0 .+ sigma*U
    Acceptance_Prob_Vec = zeros(N)
    D = -1*ones(N)
    for i = 1:N
        Acceptance_Prob_Vec[i]= exp(min(0,logPrior(NewP[i,:])-logPrior(P0[i,:])+log(dist(NewP[i,:],z=Z[i,:])<epsilon)))
        u = rand(Uniform(0,1))
        if u>= Acceptance_Prob_Vec[i]
            NewP[i,:] = P0[i,:]
        else
            D[i] = dist(NewP[i,:],z=Z[i,:])
        end
    end
    return (NewParticles = NewP,Average_Accept_Rate=mean(Acceptance_Prob_Vec),D=D,alpha=Acceptance_Prob_Vec)
end

function LocalMH(P0,epsilon,rang;Σ,dist,no_obs)
    N,C = size(P0)
    U = zeros(N,C)
    for i = 1:N
        U[i,:] = rand(MultivariateNormal(zeros(C),Σ))
    end
    Z = rand(Normal(0,1),N,no_obs)
    object(x) = -ESJD_NaiveABC(x,P0=P0,U=U,Z=Z,epsilon=epsilon,dist=dist)
    opt_sigma = optimize(object,rang[1],rang[2]).minimizer
    NewP,AcceptRate,Dist,_ = MHUpdate_NaiveABC(opt_sigma,P0=P0,U=U,Z=Z,epsilon=epsilon,dist=dist)
    return (Opt_Sigma = opt_sigma,NewP=NewP,Dist=Dist,AcceptRate=AcceptRate)
end

function NaiveABC(N,T,y;Threshold,rang)
    no_obs = length(y)
    P      = zeros(N,4,T+1)
    W      = zeros(N,T+1)
    A      = zeros(Int64,N,T)
    alpha  = zeros(T)

    D = zeros(N,T+1)
    ϵ = zeros(T+1)
    SVec = zeros(T)
    d(θ;z) = norm(f.(z,θ=θ) .- y)
    for i = 1:N
        P[i,:,1] = rand(Uniform(0,10),4)
        z = rand(Normal(0,1),no_obs)
        D[i,1] = d(P[i,:,1],z=z)
    end
    ϵ[1]=findmax(D[:,1])[1]
    W[:,1] .= 1/N;

    @showprogress 1 "Computing.." for t = 1:T
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...);
        if length(unique(D[:,t]))<500
            ϵ[t+1] = ϵ[t]
        else
            ϵ[t+1] = sort(unique(D[:,t]))[500]
        end
        
        W[:,t+1] = (D[A[:,t],t] .< ϵ[t+1])/sum(D[A[:,t],t] .< ϵ[t+1])
        Σ = cov(P[A[:,t],:,t])+Diagonal(repeat([1e-8],4));
        SVec[t],P[:,:,t+1],D[:,t+1],alpha[t]=LocalMH(P[A[:,t],:,t],ϵ[t+1],rang,Σ=Σ,dist=d,no_obs=no_obs)
        D[findall(D[:,t+1].<0),t+1] = D[A[findall(D[:,t+1].<0),t],t]
    end
    return (Particles = P, Ancestors = A, Weights = W, Epsilon = ϵ, Distance = D, OptimalScale=SVec,alpha=alpha)
end

R = NaiveABC(1000,200,y0,Threshold=0.9,rang=[0.0,5.0])

Random.seed!(123);
θ0 = [3.0,1.0,2.0,0.5];
y0 = f.(rand(Normal(0,1),20),θ=θ0)

density(R.Particles[:,4,end])

findunique(x) = length(unique(x))

plot(mapslices(findunique,R.Distance,dims=1)[1,:])

plot(R.alpha)

mapslices(findunique,R.Distance,dims=1)

R2 = SMC_RW(1000,500,y0,Threshold=0.9,rang=[0.0,5.0])

plot(log.(R.Epsilon))
plot!(log.(R2.Epsilon))

plot(mapslices(findunique,R.Distance,dims=1)[1,:])
plot!(mapslices(findunique,R2.Distance,dims=1)[1,:])

density(R.Particles[:,4,end]);density!(R2.Particles[:,4,end])

log.(R.Epsilon)

log.(R2.Epsilon)

density(R.Particles[:,1,end])

density(R2.Particles[:,10,end])

