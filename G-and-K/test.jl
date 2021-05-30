using Plots,StatsPlots, JLD2, LinearAlgebra
include("main.jl")
include("MCMC/MCMC.jl")

BPSSMC_K3_Ordered = BPS.SMC(1000,400,dat20,Threshold=0.9,δ=0.5,κ=3.0,K0=10,MH=BPS.BPS1,Dist=BPS.Dist1)
BPS.epdf(BPSSMC_K4_Ordered,3,151,color=:darkolivegreen)