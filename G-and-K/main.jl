using Plots, StatsPlots, Distributions, TimerOutputs, LinearAlgebra
include("src/RE-SMC.jl")
include("src/RW.jl")
include("src/utils.jl")

θ0 = [3.0,1.0,2.0,0.5]
ystar = utils.DataGenerator(θ0,20,12345)

ϵvec = [1.0,2.0,3.0,5.0,10.0,15.0,20.0,25.0]
RW_Particles = Array{Matrix,1}(undef,length(ϵvec))
RWCostVec    = zeros(length(ϵvec))
for n = 1:length(ϵvec)
    R = RW.SMC(10000,ystar,η=0.9,TerminalTol=ϵvec[n])
    RW_Particles[n] = utils.transferTheta(R,length(R.U))
    RWCostVec[n]    = log(utils.MCStepPerESS(R))
end

RE_Particles = Array{Matrix,1}(undef,length(ϵvec))
RECostVec    = zeros(length(ϵvec))
for n = 1:length(ϵvec)
    R = RESMC.PMMH(θ0,2000,5000,y=ystar,ϵ=ϵvec[n],Σ=cov(RW_Particles[n]))
    RE_Particles[n] = R.theta
    RECostVec[n]    = log(sum(R.NumVec)/utils.ESS(R.theta))
end


plot(ϵvec,RWCostVec,xlabel="Tolerance Level",ylabel="Log(MCMC steps/ESS)",label="RW-ABC-SMC");
scatter!(ϵvec,RWCostVec,label="",markersize=5.0,color=:white);
plot!(ϵvec,RECostVec,label="RE-ABS-SMC");
scatter!(ϵvec,RECostVec,label="",markersize=5.0,color=:white)

PLOTS = Array{Any,1}(undef,6)
for i = 1:6
    PLOTS[i] = plot(RE_Particles[i][:,1],label="epsilon=$(ϵvec[i])",xlabel="Iteration",ylabel="a")
end
plot(PLOTS[1],PLOTS[2],PLOTS[3],PLOTS[4],PLOTS[5],PLOTS[6],layout=(2,3),size=(900,600))


v = @timed R = RW.SMC(5000,ystar,η=0.9,TerminalTol=10.0,GarbageCollect=false)
NC(R) = sum(log.(mapslices(ESS,R.WEIGHT,dims=1) / 5000))

NCvec = zeros(50)
for n = 1:50
    R = RW.SMC(5000,ystar,η=0.8,TerminalTol=10.0,GarbageCollect=false)
    NCvec[n] = NC(R)
end
plot(NCvec)
density(NCvec)

X = utils.transferTheta(R,length(R.U))

plot(X[:,3])

density(X[:,1])
