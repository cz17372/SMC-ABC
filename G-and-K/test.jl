include("main.jl")
i
R = BPS.BPS_SMC_ABC(1000,50,dat20,Threshold=0.8,δ=0.5,κ=0.5,K0=10)

using Plots,StatsPlots, JLD2

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/RW_Thres075_Step01.jld2"
RW_Thres075_Step01 = R_RW
@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/RW_Thres080_Step01.jld2"
RW_Thres080_Step01 = R_RW

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-UnorderNormMetric/100obs/RW_Thres075_Step015_100.jld2"
RW_Thres080_Step015_100 = R_RW

@load "C:/Users/chang/OneDrive - University of Bristol/SMC-Data/Random-Walk/RW-OrderNormMetric/100Particles/RW_Thres070_Step01_100.jld2"
RW_Thres080_Step010_100_Ordered = R_RW

RandomWalk.Epsilon(RW_Thres080_Step010_100_Ordered,color=:darkolivegreen)