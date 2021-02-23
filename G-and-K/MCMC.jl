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
    return find_zero(z->f(z;θ=θ)-x,(lower,upper))
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
    return (Sample=Output,AcceptanceProb=AcceptanceProb/(N-1))
end
function RWMPlot(Chain,Burnin,VarName,TrueVar)
    p1 = plot(Chain[:,1],label="",linewidth=0.5,xlabel="Iteration",ylabel=VarName[1])
    hline!([TrueVar[1]],label="")
    p2 = density(Chain[Burnin:end,1],label="",xlabel=VarName[1],ylabel="Density")
    vline!([TrueVar[1]],label="")
    p3 = plot(Chain[:,2],label="",linewidth=0.5,xlabel="Iteration",ylabel=VarName[2])
    hline!([TrueVar[2]],label="")
    p4 = density(Chain[Burnin:end,2],label="",xlabel=VarName[2],ylabel="Density")
    vline!([TrueVar[2]],label="")
    p5 = plot(Chain[:,3],label="",linewidth=0.5,xlabel="Iteration",ylabel=VarName[3])
    hline!([TrueVar[3]],label="")
    p6 = density(Chain[Burnin:end,3],label="",xlabel=VarName[3],ylabel="Density")
    vline!([TrueVar[3]],label="")
    p7 = plot(Chain[:,4],label="",linewidth=0.5,xlabel="Iteration",ylabel=VarName[4])
    hline!([TrueVar[4]],label="")
    p8 = density(Chain[Burnin:end,4],label="",xlabel=VarName[4],ylabel="Density")
    vline!([TrueVar[4]],label="")
    plot(p1,p2,p3,p4,p5,p6,p7,p8,layout=(2,4),size=(800,400))
end