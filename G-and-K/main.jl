using Plots, StatsPlots, Distributions
include("src/RE-SMC.jl")
include("src/RW.jl")
include("src/utils.jl")

θ0 = [3.0,1.0,2.0,0.5]
ystar = utils.DataGenerator(θ0,20,12345)


R = RW.SMC(10000,ystar,η=0.9,TerminalTol=1.0)
Σ =  cov(utils.transferTheta(R,length(R.U)))
#plot the data
theme(:dark)
plt_data = plot(ystar,label="",color=:yellow,size=(500,500),xlabel="data index",ylabel="observations")

RareEvent = RESMC.PMMH(θ0,1000,2000;y=ystar,ϵ=1.0,Σ = Σ)

utils.ESS(RareEvent.theta)
