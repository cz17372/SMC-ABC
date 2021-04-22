using Distributions, Plots, LinearAlgebra
using ForwardDiff: gradient
Σ = [1 2;2 5]
μ = [0,0]
C(x) = norm(x) - 2
Bgrad(x) = gradient(C,x)

function logpi(x)
    if C(x) < 0
        return logpdf(MultivariateNormal(μ,Σ),x)
    else
        return -Inf
    end
end
Egrad(x) = gradient(u->-logpdf(MultivariateNormal(μ,Σ),u),x)
α1(x,u,δ) = min(0,logpi(x+δ*u)-logpi(x))
φ1(x,u,δ) = (x+δ*u,-u)
R(u,x) = u - 2 * transpose(u) * normalize(Egrad(x)) * normalize(Egrad(x))
b(x,u) = (x,R(u,x))
function φ2(x,u,δ)
    x1,u1 = φ1(x,u,δ)
    x2,u2 = b(x1,u1)
    return φ1(x2,u2,δ)
end
Rb(u,x) = u - 2*transpose(u)*normalize(Bgrad(x)) * normalize(Bgrad(x))
bb(x,u) = (x,Rb(u,x))
function φ2(x,u,δ)
    x1,u1 = φ1(x,u,δ)
    x2,u2 = b(x1,u1)
    return φ1(x2,u2,δ)
end
function φ2b(x,u,δ)
    x1,u1 = φ1(x,u,δ)
    x2,u2 = bb(x1,u1)
    return φ1(x2,u2,δ)
end
function α2(x,u,δ)
    x2,u2 = φ2(x,u,δ)
    first_rejection_ratio = log(1-exp(α1(x2,u2,δ))) - log(1-exp(α1(x,u,δ)))
    llk_ratio = logpi(x2) - logpi(x)
    return min(0,first_rejection_ratio + llk_ratio)
end
function α2b(x,u,δ)
    x2,u2 = φ2b(x,u,δ)
    llk_ratio = logpi(x2) - logpi(x)
    return min(0,llk_ratio)
end


function BPS(N,x0,u0,δ)
    X = zeros(N,2)
    X[1,:] = x0
    u0 = normalize(rand(Normal(0,1),2))
    for n = 2:N
        x1,u1 = φ1(X[n-1,:],u0,δ)
        if C(x1) >= 0
            # Propose the boundary reflection
            x2,u2 = φ2b(X[n-1,:],u0,δ)
            alpha2 = α2b(X[n-1,:],u0,δ)
            if log(rand(Uniform(0,1))) < alpha2
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = -u0
            end
        else
            alpha1 = α1(X[n-1,:],u0,δ)
        end
    end
end
