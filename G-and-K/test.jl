using Plots,StatsPlots, JLD2, LinearAlgebra
include("main.jl")
include("MCMC/MCMC.jl")

MCMC,α = RWM(50000,1.0*I,0.2,y=dat20)
# 20 data, sorted distance metric
BPSSMC_K4 = BPS.SMC(1000,300,dat20,Threshold=0.8,δ=0.5,κ=13.0,K0=10,MH=BPS.BPS1,Dist=BPS.Dist2)
BPSSMC_K10_Ordered = BPS.SMC(1000,200,dat20,Threshold=0.8,δ=0.5,κ=10.0,K0=10,MH=BPS.BPS1,Dist=BPS.Dist1)
BPSSMC_K5_Ordered = BPS.SMC(1000,200,dat20,Threshold=0.8,δ=0.5,κ=5.0,K0=10,MH=BPS.BPS1,Dist=BPS.Dist1)
BPSSMC_K4_Ordered = BPS.SMC(1000,150,dat20,Threshold=0.8,δ=0.5,κ=2.0,K0=10,MH=BPS.BPS1,Dist=BPS.Dist1)
BPS.epdf(BPSSMC_K5_Ordered,4,151,color=:darkolivegreen)