using ProgressMeter: Distributed
using Plots: get_aspect_ratio
using LinearAlgebra, Distributions
using ForwardDiff: gradient
using ProgressMeter
using Plots, StatsPlots, Random
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
# Defines the boundary for constrained region, parameterized by ϵ
Euclidean(x;y) = norm(f.(x[5:end],θ=x[1:4]) .- y)
C(x;y,ϵ,Dist)  = Dist(x,y=y) - ϵ

function logpi(ξ;ϵ,y,Dist)
    if C(ξ,ϵ=ϵ,y=y,Dist=Dist) < 0
        logpdf_θ = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logpdf_z = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logpdf_θ + logpdf_z
    else
        return -Inf
    end
end

# Define the energy function for internal reflection 
U(x) = sum(logpdf.(Uniform(0,10),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))

function φ1(x0::Vector{Float64},u0::Vector{Float64},δ::Float64)
    return (x0 .+ δ*u0, -u0)
end

function EnergyBounce(x0,u0;gradFunc)
    n = normalize(gradient(gradFunc,x0))
    return u0 .- 2.0*dot(u0,n)*n
end

function BoundaryBounce(x0,u0;gradFunc)
    if any([(x0[1:4].<0);(x0[1:4].>10)])
        n = normalize(1.0*(x0 .< 0) .- 1.0*(x0 .> 10))
    else
        n = normalize(gradient(gradFunc,x0))
    end
    return u0 .- 2.0*dot(u0,n)*n
end

σ(x0,u0) = (x0,-u0)

function φ2(x0,u0,δ;BounceType,gradFunc)
    x1, u1 = φ1(x0,u0,δ)
    xflip,uflip = σ(x1,u1)
    xbounce = xflip; ubounce = BounceType(xflip,uflip,gradFunc=gradFunc)
    x2,u2 = φ1(xbounce,ubounce,δ)
    return x2,u2
end

function get_Cov_Matrix(x,η;y,ϵ,Dist)
    n = normalize(gradient(a->C(a,y=y,ϵ=ϵ,Dist=Dist),x))
    A = I - η*n*transpose(n)
    return A*transpose(A)/length(x)
end

Random.seed!(123)
z0 = rand(Normal(0,1),20)
θ0 = [3.0,1.0,2.0,0.5]
dat20 = f.(z0,θ=θ0)
x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
while Euclidean(x0,y=dat20) > 5.0
    x0 = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
end


dist_vec = zeros(10000)
Σ =  get_Cov_Matrix(x0,0.9,y=dat20,ϵ=5.0,Dist=Euclidean)
u0 = rand(MultivariateNormal(zeros(24),Σ))
x1 = x0 .+ 0.3*u0
for i = 1:10000
    u0 = rand(MultivariateNormal(zeros(24),Σ))
    x1 = x0 .+ 0.3*u0
    dist_vec[i] = Euclidean(x1,y=dat20)
end
std(dist_vec)
logpdf(MultivariateNormal(zeros(24),Σ),u0)
Euclidean(x1,y=dat20)
Σ2 = get_Cov_Matrix(x1,0.9,y=dat20,ϵ=0.5,Dist=Euclidean)
logpdf(MultivariateNormal(zeros(24),Σ2),-u0)

plot(dist_vec)

dist_vec2 = zeros(10000)
for i = 1:10000
    u0 = rand(MultivariateNormal(zeros(24),1.0/24*I))
    x1 = x0 .+ 0.3*u0
    dist_vec2[i] = Euclidean(x1,y=dat20)
end
mean(dist_vec2 .< 5.0)
mean(dist_vec .< 5.0)
plot(dist_vec2,label="Isotropic")
plot!(dist_vec,label="Scale = 0.9")
hline!([5.0],label="",color=:red,linewidth=3.0)
