using Distributions, Roots, LinearAlgebra, Random
using ForwardDiff: derivative
function f(z;θ)
    return θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
end
function invz(x;g)
    lower=0.0
    upper= 0.0
    while (g(lower)-x)*(g(upper)-x) >= 0
        lower -= 1
        upper += 1
    end
    return find_zero(z->g(z)-x,(lower,upper))
end
function gkpdf(θ;x)
    g(x) = f(x,θ=θ)
    z = invz.(x,g=g)
    return sum(log.(pdf.(Normal(0,1),z) ./ abs.(derivative.(Ref(g),z))))
end
logPrior(θ) = sum(logpdf.(Uniform(0,10),θ))
function MCMC(N,y0,init_theta,cov_proposal)
    ThetaMat = zeros(N,length(init_theta))
    ThetaMat[1,:] = init_theta
    llk(θ) = gkpdf(θ,x=y0)
    @showprogress 1 "Computing..." for t = 2:N
        theta_star = rand(MultivariateNormal(ThetaMat[t-1,:],0.5^2*cov_proposal))
        logalpha = min(0,logPrior(theta_star)-logPrior(ThetaMat[t-1,:])+llk(theta_star)-llk(ThetaMat[t-1,:]))
        logu = log(rand(Uniform(0,1)))
        if logu < logalpha
            ThetaMat[t,:] = theta_star
        else
            ThetaMat[t,:] = ThetaMat[t-1,:]
        end
    end
    return ThetaMat
end

