using Distributions, LinearAlgebra, ProgressMeter,Plots, StatsPlots, Random, Flux

function Transform_Normal(z;par)
    # Define the G-and-K model using standar Normal distributions
    return par[1] + par[2]*(1+0.8*(1-exp(-par[3]*z))/(1+exp(-par[3]*z)))*((1+z^2)^par[4])*z
end

function φ(ξ)
    θ = ξ[1:4]
    z = ξ[5:end]
    return Transform_Normal.(z,par=θ)
end    

function NewSMC(N,T,data;Method="RW",σ=0.05,Threshold=0.8)
    Particles = zeros(N,4 + length(data),T+1)
    ϵ = zeros(T+1)
    W = zeros(N,T+1)
    A = zeros(Int64,N,T)
    Distance = zeros(N,T+1)
    function Transform_Normal(z;par)
        # Define the G-and-K model using standar Normal distributions
        return par[1] + par[2]*(1+0.8*(1-exp(-par[3]*z))/(1+exp(-par[3]*z)))*((1+z^2)^par[4])*z
    end
    function φ(ξ)
        θ = ξ[1:4]
        z = ξ[5:end]
        return Transform_Normal.(z,par=θ)
    end
    function d(ξ;y=data)
        x = φ(ξ)
        return mean((y .- x).^2)
    end
    function logPrior(ξ)
        logparam = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logz     = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logparam + logz
    end
    function LocalMH(ξ0,Σ,epsilon,Method,sigma)
        if Method == "RW"
            newξ = rand(MultivariateNormal(ξ0,sigma*Σ))
            u = rand(Uniform(0,1))
            if log(u) >= logPrior(newξ)-logPrior(ξ0)
                return ξ0
            else
                x = φ(newξ)
                if d(newξ) < epsilon
                    return newξ
                else
                    return ξ0
                end
            end
        elseif Method == "Langevin"
            forward_d = MultivariateNormal(ξ0 .- sigma^2*gradient(d,ξ0)[1],sigma^2*I)
            newξ = rand(forward_d)
            backward_d = MultivariateNormal(newξ .- sigma^2*gradient(d,newξ)[1],sigma^2*I)
            log_q = logpdf(backward_d,ξ0) - logpdf(forward_d,newξ)
            logprior=logPrior(newξ)-logPrior(ξ0)
            u = rand(Uniform(0,1))
            if u >= log_q + logprior
                return ξ0
            else
                x = φ(newξ)
                if d(newξ) < epsilon
                    return newξ
                else
                    return ξ0
                end
            end
        end
    end
    t = 0
    for n = 1:N
        Particles[n,:,t+1] = [rand(Uniform(0,10),4);rand(Normal(0,1),length(data))]
        Distance[n,t+1] = d(Particles[n,:,t+1])
    end
    ϵ[t+1] = findmax(Distance[:,t+1])[1]
    W[:,t+1] .= 1/N;
    @showprogress 1 "Computing..." for t = 1:T
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...)
        ϵ[t+1] = sort(Distance[A[:,t],t])[floor(Int,Threshold*N)]
        W[:,t+1] = (Distance[:,t] .< ϵ[t+1])/sum(Distance[:,t] .< ϵ[t+1])
        Σ = cov(Particles[A[:,t],:,t])
        for n = 1:N
            Particles[n,:,t+1] = LocalMH(Particles[A[n,t],:,t],Σ,ϵ[t+1],Method,σ)
            Distance[n,t+1] = d(Particles[n,:,t+1])
        end
    end
    return (P=Particles,W=W,A=A,epsilon=ϵ,D=Distance)
end
