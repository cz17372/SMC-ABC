using Flux, Distributions
using Flux: params
using Random, StatsPlots, Plots
using LinearAlgebra
using ProgressMeter

function f(z,θ)
    return θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
end 

function φ(ξ)
    θ = ξ[1:4]
    z = ξ[5:end]
    return f.(z,Ref(θ))
end

function unit(x)
    return x / norm(x)
end

θ0 = [3.0,1.0,2.0,0.5];
Random.seed!(123);
z0 = rand(Normal(0,1),20);
y0 = φ([θ0;z0]);

function unit(x)
    return x / norm(x)
end

function dist(ξ;y=y0)
    x = φ(ξ)
    return norm(y .- x)
end

function logPrior(ξ)
    θ = ξ[1:4]
    z = ξ[5:end]
    return sum(logpdf.(Uniform(0,10),θ)) + sum(logpdf.(Normal(0,1),z))
end

function LocalMH(ξ0,Σ,ϵ;Method=Method,sigma=σ)
    if Method == "RW" # A random walk proposal is used in this case
        newξ = rand(MultivariateNormal(ξ0,sigma^2*Σ))
        prior_ratio = logPrior(newξ) - logPrior(ξ0)
        logα = min(0,prior_ratio+log(dist(newξ)<ϵ))
        u = rand(Uniform(0,1))
        if log(u) < logα
            return (newξ,exp(logα))
        else
            return (ξ0,exp(logα))
        end
    end
    if Method == "Langevin"
        forward_distr = MultivariateNormal(ξ0 .- sigma^2 * unit(gradient(dist,ξ0)[1]),sigma^2*Σ)
        newξ  = rand(forward_distr)
        backward_distr = MultivariateNormal(newξ .- sigma^2 * unit(gradient(dist,newξ)[1]),sigma^2*Σ)
        u = rand(Uniform(0,1))
        prop_ratio = logpdf(backward_distr,ξ0) - logpdf(forward_distr,newξ)
        prior_ratio = logPrior(newξ) - logPrior(ξ0)
        logα = min(0,prop_ratio+prior_ratio+log(dist(newξ)<ϵ))
        if log(u) < logα
            return (newξ,exp(logα))
        else
            return (ξ0,exp(logα))
        end
    end
end

function NewSMC(N,T,data;Method = "RW", Threshold=0.8, σ = 0.3)
    function f(z,θ)
        return θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
    end 
    
    function φ(ξ)
        θ = ξ[1:4]
        z = ξ[5:end]
        return f.(z,Ref(θ))
    end

    function unit(x)
        return x / norm(x)
    end

    function dist(ξ;y=data)
        x = φ(ξ)
        return norm(y .- x)
    end

    function logPrior(ξ)
        θ = ξ[1:4]
        z = ξ[5:end]
        return sum(logpdf.(Uniform(0,10),θ)) + sum(logpdf.(Normal(0,1),z))
    end

    function LocalMH(ξ0,Σ,ϵ;Method=Method,sigma=σ)
        if Method == "RW" # A random walk proposal is used in this case
            newξ = rand(MultivariateNormal(ξ0,sigma^2*Σ))
            prior_ratio = logPrior(newξ) - logPrior(ξ0)
            logα = min(0,prior_ratio+log(dist(newξ)<ϵ))
            u = rand(Uniform(0,1))
            if log(u) < logα
                return (newξ,exp(logα))
            else
                return (ξ0,exp(logα))
            end
        end
        if Method == "Langevin"
            forward_distr = MultivariateNormal(ξ0 .- sigma^2 * unit(gradient(dist,ξ0)[1]),sigma^2*Σ)
            newξ  = rand(forward_distr)
            backward_distr = MultivariateNormal(newξ .- sigma^2 * unit(gradient(dist,newξ)[1]),sigma^2*Σ)
            u = rand(Uniform(0,1))
            prop_ratio = logpdf(backward_distr,ξ0) - logpdf(forward_distr,newξ)
            prior_ratio = logPrior(newξ) - logPrior(ξ0)
            logα = min(0,prop_ratio+prior_ratio+log(dist(newξ)<ϵ))
            if log(u) < logα
                return (newξ,exp(logα))
            else
                return (ξ0,exp(logα))
            end
        end
    end

    ϵ = zeros(T+1);
    W = zeros(N,T+1);
    A = zeros(Int64,N,T+1);
    D = zeros(N,T+1)
    P = zeros(N,4+length(data),T+1);
    t = 0;
    acc_mat = zeros(N,T);
    for n = 1:N
        P[n,:,t+1] = [rand(Uniform(0,10),4);rand(Normal(0,1),length(data))]
        D[n,t+1] = dist(P[n,:,t+1])
    end
    ϵ[t+1] = findmax(D[:,t+1])[1]
    W[:,t+1] .= 1/N;
    @showprogress 1 "Computing..." for t = 1:T
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...)
        ϵ[t+1] = sort(D[A[:,t],t])[floor(Int64,Threshold*N)]
        W[:,t+1] = (D[:,t] .< ϵ[t+1])/sum(D[:,t] .< ϵ[t+1])
        #Σ = cov(P[A[:,t],:,t])
        Σ = 1.0 * I;
        for n = 1:N
            xi,alpha = LocalMH(P[A[n,t],:,t],Σ,ϵ[t+1],Method=Method)
            P[n,:,t+1] = xi
            acc_mat[n,t] = alpha
            D[n,t+1] = dist(P[n,:,t+1])
        end
    end
    return (P=P,W=W,A=A,epsilon=ϵ,D=D,α = acc_mat)
end

R = NewSMC(10000,500,y0,Method = "Langevin",σ=0.05)
R2 = NewSMC(10000,500,y0,σ = 0.1)
plot(mean(R.α,dims=1)[1,:],xlabel="Iteration",ylabel="Average acceptance probability",label="Langevin Update")
plot!(mean(R2.α,dims=1)[1,:],label="RW")



plot(log.(R.epsilon))
plot!(log.(R2.epsilon))