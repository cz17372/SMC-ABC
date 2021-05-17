include("main.jl")
using Plots,StatsPlots, JLD2, LinearAlgebra
R = BPS.SMC(1000,100,dat20,Threshold=0.8,δ=0.5,κ=0.7,K0=10,MH=BPS.BPS2)
index = findall(R.DISTANCE[:,end] .> 0)
density(R.U[1,index,end])
plot(log.(R.EPSILON))


@load "C:/Users/chang/OneDrive/Documents/BPS2_Thres080_Stepsize045_Refresh060.jld2"
R = BPS2_Thres080_Stepsize045_Refresh060

plot(log.(R.EPSILON))
index = findall(R.DISTANCE[:,end] .> 0)
density(R.U[4,index,end])
R.BoundaryBounceProposed