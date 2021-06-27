using JLD2, Plots, StatsPlots, Distributions, Random, LinearAlgebra
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
Random.seed!(123)
z0 = rand(Normal(0,1),20)
θ0 = [3.0,1.0,2.0,0.5]
dat20 = f.(z0,θ=θ0)
cd("G-and-K"); 
include("utils.jl"); include("MCMC/MCMC.jl")

MCMC, α = RWM(1000000,1.0*I,0.2,y=dat20)
Σ = cov(MCMC)
plot(MCMC[:,1])
density(MCMC[500001:end,4])
n = 4
xlim=(-0.5,3.0)
@load "data/20data_RW_1000Particles_Fixed.jld2"
data20_1000Particles_RW_Fixed = Results
@load "data/20data_RW_1000Particles_Adaptive.jld2"
data20_1000Particles_RW_Adaptive = Results
p1 = plotposterior(data20_1000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 1000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p2 = plotposterior(data20_1000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 1000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
@load "data/20data_RW_2000Particles_Fixed.jld2"
data20_2000Particles_RW_Fixed = Results
@load "data/20data_RW_2000Particles_Adaptive.jld2"
data20_2000Particles_RW_Adaptive = Results
p3 = plotposterior(data20_2000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 2000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p4 = plotposterior(data20_2000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 2000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
@load "data/20data_RW_5000Particles_Fixed.jld2"
data20_5000Particles_RW_Fixed = Results
@load "data/20data_RW_5000Particles_Adaptive.jld2"
data20_5000Particles_RW_Adaptive = Results
p5 = plotposterior(data20_5000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 5000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p6= plotposterior(data20_5000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 5000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
@load "data/20data_RW_10000Particles_Fixed.jld2"
data20_10000Particles_RW_Fixed = Results
@load "data/20data_RW_10000Particles_Adaptive.jld2"
data20_10000Particles_RW_Adaptive = Results
p7 = plotposterior(data20_10000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 10000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p8= plotposterior(data20_10000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 10000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
@load "data/20data_BPS_1000Particles_Adaptive.jld2"
data20_1000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_2000Particles_Adaptive.jld2"
data20_2000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_5000Particles_Adaptive.jld2"
data20_5000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_10000Particles_Adaptive.jld2"
data20_10000Particles_BPS_Adaptive = Results
p9 = plotposterior(data20_1000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 1000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p10 = plotposterior(data20_2000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 2000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p11 = plotposterior(data20_5000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 5000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
p12 = plotposterior(data20_10000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 10000 Particles",xlim=xlim);
density!(MCMC[500001:end,n],label="",color=:red,linewidth=2.0);
plot(p1,p3,p5,p7,p2,p4,p6,p8,p9,p10,p11,p12,layout=(3,4),size=(2400,1800))



p1 = plotK(data20_1000Particles_BPS_Adaptive,label="BPS-Adaptive Stepsize");
plotK(data20_1000Particles_RW_Fixed,label="RW-Fixed Stepsize",color=:red,new=false,xlabel="SMC Iteration",ylabel="No. MCMC steps");
plotK(data20_1000Particles_RW_Adaptive,title="1000 Particles",label="RW-Adaptive Stepsize",color=:darkolivegreen,new=false,xlabel="SMC Iteration",ylabel="log No. MCMC steps");

p2 = plotK(data20_2000Particles_BPS_Adaptive,label="BPS-Adaptive Stepsize");
plotK(data20_2000Particles_RW_Fixed,label="RW-Fixed Stepsize",color=:red,new=false,xlabel="SMC Iteration",ylabel="No. MCMC steps");
plotK(data20_2000Particles_RW_Adaptive,title="2000 Particles",label="RW-Adaptive Stepsize",color=:darkolivegreen,new=false,xlabel="SMC Iteration",ylabel="log No. MCMC steps");

p3 = plotK(data20_5000Particles_BPS_Adaptive,label="BPS-Adaptive Stepsize");
plotK(data20_5000Particles_RW_Fixed,label="RW-Fixed Stepsize",color=:red,new=false,xlabel="SMC Iteration",ylabel="No. MCMC steps");
plotK(data20_5000Particles_RW_Adaptive,title="5000 Particles",label="RW-Adaptive Stepsize",color=:darkolivegreen,new=false,xlabel="SMC Iteration",ylabel="log No. MCMC steps");

p4 = plotK(data20_10000Particles_BPS_Adaptive,label="BPS-Adaptive Stepsize");
plotK(data20_10000Particles_RW_Fixed,label="RW-Fixed Stepsize",color=:red,new=false,xlabel="SMC Iteration",ylabel="No. MCMC steps");
plotK(data20_5000Particles_RW_Adaptive,title="10000 Particles",label="RW-Adaptive Stepsize",color=:darkolivegreen,new=false,xlabel="SMC Iteration",ylabel="log No. MCMC steps");

plot(p1,p2,p3,p4,layout=(2,2),size=(1200,1200))

savefig("20data-gandk-NoMCMCSteps.pdf")

include("BPS/ExactBPS-SMC-ABC.jl")

R = ExactBPS.SMC(1000,250,dat20,Threshold=0.8,δ=0.3,κ=3.0,K0=2,MaxBounces=2.0,MinStepsize=0.1)


@load "data/20data_Langevin_2000Particles_Adaptive.jld2"

n = 3; xlim=(0.0,11.0)
plotposterior(Results,n,xlim=xlim)

@load "Experiment/try.jld2"
n = 3; xlim=(0.0,11.0)
plotposterior(Results,n,xlim=xlim)

@load "Experiment/try.jl2"

cd("G-and-K")
R = RandomWalk.SMC(1000,250,dat20,Threshold=0.8,δ=0.3,K0=5,MinAcceptProbability=0.25,MinStepSize=0.1)

x0 = R.U[:,1,end]

index = findall(R.WEIGHT[:,end] .> 0)
U = R.U[:,index,end]

Euclidean(x0,y=dat20)
