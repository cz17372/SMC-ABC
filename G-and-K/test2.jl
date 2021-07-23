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


