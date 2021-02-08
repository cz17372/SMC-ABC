include("G-and-K/MCMC.jl")

using JLD2, FileIO
using Plots, StatsPlots
theme(:mute)

@load "G-and-K/data.jld2" y0
ystar = y0;


#RWM_Σ = 1.0*I
RWM_Σ = cov(RWM_RES[20001:end,:])
@save "RWM_COV.jld2" RWM_Σ
RWM_RES,RWM_ACCEPTANCE_RATE = RWM(100000,RWM_Σ,0.2)

RWMPlot(RWM_RES,20001,["a","b","g","k"],[3.0,1.0,2.0,0.5])
