using Distributions, Plots, StatsPlots, Random, LinearAlgebra, JLD2
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function g(x;θ)
    z = quantile(Normal(0,1),x)
    return f.(z,θ=θ)
end
function r(y)
    m = rem(y,2)
    if m < 0
        m += 2
    end
    if m < 1
        return m
    else
        return 2 - m
    end
end
function SliceSampling(x0;ϵ,ϕ,w)
    # sample velocity
    v = rand(Normal(0,1),length(x0))
    u = rand(Uniform(0,w)); a = -u; b= w-u
    NoSteps = 0
    while true
        z = rand(Uniform(a,b))
        x1 = r.(x0 .+ z*v)
        NoSteps += 1
        if ϕ(x1) < ϵ
            return (x1,NoSteps,abs(z))
        else
            if z < 0.0
                a = z
            else
                b = z
            end
        end
    end
end

