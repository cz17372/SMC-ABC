using Distributions, Random, LinearAlgebra, Plots, StatsPlots
include("src/RW.jl")
include("src/RESMC.jl")
include("src/gkn.jl")

θstar = [3.0,1.0,2.0,0.5]
Random.seed!(123)
ystar = gkn.ConSimulator(20,θstar)

Dist(x,y) = norm(x .- y)
R = RW.SMC(10000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=0.2,η = 0.8,gc=false)

Index = findall(R.WEIGHT[:,end] .> 0)
U = R.U[end][:,findall(R.WEIGHT[:,end] .> 0)]
transform(x) = 10*cdf(Normal(0,1),x)
X = mapslices(transform,U[1:4,:],dims=2)

density(X[1,:])

XArray = Array{Matrix,1}(undef,20)
for i = 1:20
    R = RW.SMC(10000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=0.2,η = 0.9,gc=false)
    U = R.U[end][:,findall(R.WEIGHT[:,end] .> 0)]
    XArray[i] = 10*cdf(Normal(0,1),U)
    GC.gc()
end
n = 4
density(XArray[1][n,:],label="",color=:grey)
for i = 2:20
    density!(XArray[i][n,:],label="",color=:grey)
end
current()