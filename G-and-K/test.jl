using JLD2, Plots, StatsPlots
cd("G-and-K")
include("utils.jl")
n = 4
xlim=(-0.5,3.0)
@load "data/20data_RW_1000Particles_Fixed.jld2"
data20_1000Particles_RW_Fixed = Results
@load "data/20data_RW_1000Particles_Adaptive.jld2"
data20_1000Particles_RW_Adaptive = Results

p1 = plotposterior(data20_1000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 1000 Particles",xlim=xlim);
p2 = plotposterior(data20_1000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 1000 Particles",xlim=xlim);


@load "data/20data_RW_2000Particles_Fixed.jld2"
data20_2000Particles_RW_Fixed = Results
@load "20data_RW_2000Particles_Adaptive.jld2"
data20_2000Particles_RW_Adaptive = Results

p3 = plotposterior(data20_2000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 2000 Particles",xlim=xlim);
p4 = plotposterior(data20_2000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 2000 Particles",xlim=xlim);


@load "data/20data_RW_5000Particles_Fixed.jld2"
data20_5000Particles_RW_Fixed = Results
@load "data/20data_RW_5000Particles_Adaptive.jld2"
data20_5000Particles_RW_Adaptive = Results

p5 = plotposterior(data20_5000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 5000 Particles",xlim=xlim);
p6= plotposterior(data20_5000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 5000 Particles",xlim=xlim);


@load "data/20data_RW_10000Particles_Fixed.jld2"
data20_10000Particles_RW_Fixed = Results
@load "data/20data_RW_10000Particles_Adaptive.jld2"
data20_10000Particles_RW_Adaptive = Results

p7 = plotposterior(data20_10000Particles_RW_Adaptive,n,title="RW, Adaptive Stepsize, 10000 Particles",xlim=xlim);
p8= plotposterior(data20_10000Particles_RW_Fixed,n,title="RW, Fixed Stepsize, 10000 Particles",xlim=xlim);

@load "data/20data_BPS_1000Particles_Adaptive.jld2"
data20_1000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_2000Particles_Adaptive.jld2"
data20_2000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_5000Particles_Adaptive.jld2"
data20_5000Particles_BPS_Adaptive = Results
@load "data/20data_BPS_10000Particles_Adaptive.jld2"
data20_10000Particles_BPS_Adaptive = Results

p9 = plotposterior(data20_1000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 1000 Particles",xlim=xlim);
p10 = plotposterior(data20_2000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 2000 Particles",xlim=xlim);
p11 = plotposterior(data20_5000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 5000 Particles",xlim=xlim);
p12 = plotposterior(data20_10000Particles_BPS_Adaptive,n,title="BPS, Adaptive Stepsize, 10000 Particles",xlim=xlim);

plot(p1,p3,p5,p7,p2,p4,p6,p8,p9,p10,p11,p12,layout=(3,4),size=(2400,1800))

plotK(data20_10000Particles_BPS_Adaptive)

plotK(data20_10000Particles_RW_Adaptive,new=false,color=:darkolivegreen)