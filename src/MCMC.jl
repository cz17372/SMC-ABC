module MCMC
using Distributions, Roots
using ForwardDiff: derivative
using ProgressMeter
using LinearAlgebra
# f is defined to transform random variables from standard Normal distribution to samples from g-and-k distribution given the parameter θ.
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function inverse(x;θ)
    lower = 0.0; upper = 0.0;
    while (f(lower,θ=θ)-x)*(f(upper,θ=θ)-x) >=0
        lower -= 1.0; upper += 1.0
    end
    return find_zero(z->f(z,θ=θ)-x,(lower,upper))
end
function log_likelihood(x;θ)
    # Find the corresponding z of x
    z = inverse(x,θ=θ)
    # Get the log-derivative
    log_grad = log(abs(derivative(x->f(x,θ=θ),z)))
    return logpdf(Normal(0,1),z) - log_grad
end
RWM_logPrior(θ) = sum(logpdf.(Uniform(0,10),θ))
function RWM(N,Σ,σ;y=ystar,θ0=rand(Uniform(0,10),4))
    C = length(θ0)
    Output = zeros(N,C)
    Output[1,:] = θ0
    AcceptanceProb = 0
    @showprogress 1 "Computing.." for t = 2:N
        theta_proposal = rand(MultivariateNormal(Output[t-1,:],σ*Σ))
        if RWM_logPrior(theta_proposal) == -Inf
            Output[t,:] = Output[t-1,:]
        else
            log_alpha = min(0,RWM_logPrior(theta_proposal)-RWM_logPrior(Output[t-1,:]) + 
                sum(log_likelihood.(y,θ=theta_proposal))-sum(log_likelihood.(y,θ=Output[t-1,:])))
            logu = log(rand(Uniform(0,1)))
            if logu < log_alpha
                Output[t,:] = theta_proposal
                AcceptanceProb += 1
            else
                Output[t,:] = Output[t-1,:]
            end
        end
    end
    return (Sample=Output,AcceptanceProb=AcceptanceProb/(N-1))
end

end