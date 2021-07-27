using ForwardDiff: gradient, derivative
using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra

U(x) = sum(logpdf(Normal(0,1),x))

function φ1(x0::Vector{Float64},u0::Vector{Float64},δ::Float64)
    return (x0 .+ δ*u0,-u0)
end

function Bounce(x1,u1,U)
    n = normalize(gradient(U,x1))
    return u1 .- 2.0*dot(u1,n)*n
end

function α1(x0,x1,logπ)
    return min(0,logπ(x1)- logπ(x0))
end


function ϕ(u)
    θ1 = 10.0*u[1]
    θ = [θ1,1.0,2.0,0.5]
    return f.(quantile(Normal(0,1),u[2:end]),θ=θ)
end

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
Random.seed!(123)
θ0 = [3.0,1.0,2.0,0.5];
u0 = rand(1) 
z0 = quantile(Normal(0,1),u0)
ystar = f.(z0,θ=θ0)




function samp(ystar,ϵ)
    while true
        ucand = rand(Uniform(0,1),2)
        if norm(ϕ(ucand) .- ystar) < ϵ
            return ucand
        end
    end
end


u = zeros(100000,2)
Threads.@threads for n = 1:100000
    u[n,:] = samp(ystar,0.01)
end
scatter(u[:,1],u[:,2],markersize=0.01,xlims=(0,1),ylims=(0,1),label="",color=:grey,size=(500,500))

S = cov(u)
A = inv(cholesky(S).L)
V = transpose(A * transpose(u))
scatter(V[:,1],V[:,2],markersize=0.01,label="",color=:grey,size=(500,500))
cov(V)
