using Distributions: Random
using Distributions, Plots
using ForwardDiff
using Random
using LinearAlgebra
using Flux
using LinearAlgebra
function ϕ(u;θ)
    θ = exp.(θ)
    N = length(u) ÷ 2
    r0 = 100.0; f0 = 100.0; dt = 1.0; σr = 1.0; σf = 1.0; 
    rvec = zeros(N+1)
    fvec = zeros(N+1)
    rvec[1] = r0; fvec[1] = f0
    for n = 1:N
        rvec[n+1] = max(rvec[n] + dt*(θ[1]*rvec[n]-θ[2]*rvec[n]*fvec[n]) + sqrt(dt)*σr*u[2*n-1],0)
        fvec[n+1] = max(fvec[n] + dt*(θ[4]*rvec[n]*fvec[n]-θ[3]*fvec[n]) + sqrt(dt)*σf*u[2*n],0)
    end
    return [rvec[2:end];fvec[2:end]]
end
function SimulateOne(θ,N)
    u = rand(Normal(0,1),2*N)
    return ϕ(u,θ=θ)
end
function D(θ,u;y)
    return norm(ϕ(u,θ=θ) .- y)
end
function U(θ)
    return sum(logpdf.(Normal(-2.0,3.0),θ))
end
function ABC_MCMC(N,θ0,x0;y,ϵ,δ,Σ)
    d = length(x0)
    oldθ = θ0
    oldx = x0;
    Ind = 0
    for n = 2:(N+1)
        newθ = rand(MultivariateNormal(oldθ,δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newθ) - U(oldθ)
            newx = SimulateOne(newθ,d)
            if norm(newx .- y) < ϵ
                oldx = newx
                oldθ = newθ
                Ind += 1
            end
        end
    end
    return (oldθ, oldx, Ind)
end

function ABC_MCMC2(N,θ0,x0;y,ϵ,δ,Σ)
    d = length(x0) ÷ 2
    Output = zeros(N+1,length(θ0))
    Output[1,:] = θ0
    oldx = x0;
    Ind = 0
    for n = 2:(N+1)
        newθ = rand(MultivariateNormal(Output[n-1,:],δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newθ) - U(Output[n-1,:])
            newx = SimulateOne(newθ,d)
            if norm(newx .- y) < ϵ
                oldx = newx
                Output[n,:] = newθ
                Ind += 1
            else
                Output[n,:] = Output[n-1,:]
            end
        else
            println("Reject")
            Output[n,:] = Output[n-1,:]
        end
    end
    return (Output, oldx, Ind)
end

Random.seed!(17372)
θstar = log.([0.4,0.005,0.05,0.001])
ustar = rand(Normal(0,1),40)
ystar = ϕ(ustar,θ=θstar)


