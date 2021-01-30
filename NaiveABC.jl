using Distributions, LinearAlgebra, Random
using ProgressMeter
f(z;θ) = θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
d(θ;z) = norm(f.(z,θ=θ) .- y0)
logPrior(θ) = sum(logpdf.(Uniform(0,1),θ))
N = 1000
T = 100
P = zeros(N,4,T+1);
D = zeros(N,T+1);
W = zeros(N,T+1);
ϵ = zeros(T+1);
for i = 1:N
    P[i,:,1] = rand(Uniform(0,10),4)
    z = rand(Normal(0,1),20)
    D[i,1] = d(P[i,:,1],z=z)
end

Threshold=0.8
W[:,1] .= 1/N;
ϵ[1] = findmax(D[:,1])[1]
t = 1

A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...);
ϵ[t+1] = quantile(D[:,t],Threshold);
W[:,t+1] = (D[A[:,t],t] .< ϵ[t+1])/sum(D[A[:,t],t] .< ϵ[t+1])

Σ = cov(P[A[:,1],:,1])

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
    return (NewParticles = NewP,Average_Accept_Rate=mean(Acceptance_Prob_Vec),D=D)
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
    NewP,AcceptRate,Dist = MHUpdate_NaiveABC(opt_sigma,P0=P0,U=U,Z=Z,epsilon=epsilon,dist=dist)
    return (Opt_Sigma = opt_sigma,NewP=NewP,Dist=Dist,AcceptRate=AcceptRate)
end

function NaiveABC(N,T,y;Threshold,rang)
    no_obs = length(y)
    P      = zeros(N,4,T+1)
    W      = zeros(N,T+1)
    A      = zeros(Int64,N,T)


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
        ϵ[t+1] = quantile(D[:,t],Threshold);
        W[:,t+1] = (D[A[:,t],t] .< ϵ[t+1])/sum(D[A[:,t],t] .< ϵ[t+1])
        Σ = cov(P[A[:,t],:,t])+Diagonal(repeat([1e-8],4));
        SVec[t],P[:,:,t+1],D[:,t+1],_=LocalMH(P[A[:,t],:,t],ϵ[t+1],rang,Σ=Σ,dist=d,no_obs=no_obs)
        D[findall(D[:,t+1].<0),t+1] = D[A[findall(D[:,t+1].<0),t],t]
    end
    return (Particles = P, Ancestors = A, Weights = W, Epsilon = ϵ, Distance = D, OptimalScale=SVec)
end

R = NaiveABC(1000,100,y0,Threshold=0.8,rang=[0.0,5.0])

density(R.Particles[:,4,end])