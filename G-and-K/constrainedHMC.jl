using Distributions, Plots, StatsPlots, LinearAlgebra, Random
using Flux: gradient
Σ = [1 2;2 5]
E(x) = 1/2*transpose(x)*inv(Σ)*x
grad(x) = gradient(E,x)[1]
C(x) = 4 - x[1]^2 - x[2]^2
function leapfrog(x0,p0,ϵ,T)
    pos = zeros(T+3,2)
    p0 = p0 .- 0.5 * ϵ * grad(x0)
    pos[1,:] = x0
    for t = 0:T
        x0 = x0 .+ ϵ*p0
        pos[t+2,:] = x0
        if C(x0) >= 0
            p0 = p0 .- ϵ*grad(x0)
        else
            nhat = normalize(gradient(C,x0)[1])
            p0 = p0 .- 2 * (transpose(p0)*nhat)*nhat
        end
    end
    x0 = x0 .+ ϵ*p0
    pos[end,:] = x0
    p0 = p0 .- 1/2*ϵ*grad(x0)
    return (pos,x0,p0)
end
H(x,p) = logpdf(MultivariateNormal([0,0],Σ),x) + logpdf(MultivariateNormal([0,0],1.0*I),p)
function CHMC(x0,p0,N,ϵ,T)
    xmat = zeros(N,2)
    xmat[1,:] = x0
    for i = 2:N
        p1 = rand(Normal(0,1),2)
        temp = p1
        _,xstar,pstar = leapfrog(xmat[i-1,:],p1,ϵ,T)
        if C(xstar) < 0
            α = -Inf
        else
            α = min(0,H(xstar,p1)-H(xmat[i-1,:],temp))
        end
        logu = log(rand(Uniform(0,1)))
        if logu < α 
            xmat[i,:] = xstar
        else
            xmat[i,:] = xmat[i-1,:]
        end
    end
    return xmat
end
theta = collect(0:0.001:2*pi)
x0 = [0,0]; p0 = rand(Normal(0,1),2);
x = CHMC(x0,p0,1000,0.1,50)
r = 2
x2 = rand(MultivariateNormal([0,0],Σ),10000)
scatter(x2[1,:],x2[2,:],label="",color=:grey,markersize=1,size=(800,800))
scatter!(x[:,1],x[:,2],color=:green,markersize=2.0,label="")
plot!(r*cos.(theta),r*sin.(theta),label="")
density(x2[1,:])
density!(x[:,1])


f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
z = rand(Normal(0,1))
ystar = f(z,θ = [3,2,1,0.5])

function dist(u)
    y = f(u[2],θ = [3,1,2,u[1]])
    return (y-ystar)^2
end

suitable_u = zeros(50000,2)
n = 1
while n <= 50000
    ucand = [rand(Uniform(0,10)),rand(Normal(0,1))]
    if dist(ucand) < 1
        suitable_u[n,:] = ucand
        n += 1
    end
end

scatter(suitable_u[:,1],suitable_u[:,2],color=:grey,markersize=1.0)

function BPS()