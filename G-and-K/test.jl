using JLD2, Plots, StatsPlots
cd("G-and-K")
include("utils.jl")
n = 3
@load "data/20data_RW_1000Particles_Fixed.jld2"
data20_1000Particles_RW_Fixed = Results
@load "data/20data_RW_1000Particles_Adaptive.jld2"
data20_1000Particles_RW_Adaptive = Results

p1 = plotposterior(data20_1000Particles_RW_Adaptive,n,title="Adaptive Stepsize, 1000 Particles")
p2 = plotposterior(data20_1000Particles_RW_Fixed,n,title="Fixed Stepsize, 1000 Particles")


@load "20data_RW_2000Particles.jld2"
data20_2000Particles_RW_Fixed = Results
@load "20data_RW_2000Particles2.jld2"
data20_2000Particles_RW_Adaptive = Results

p3 = plotposterior(data20_2000Particles_RW_Adaptive,n,title="Adaptive Stepsize, 2000 Particles")
p4 = plotposterior(data20_2000Particles_RW_Fixed,n,title="Fixed Stepsize, 2000 Particles")


@load "20data_RW_5000Particles.jld2"
data20_5000Particles_RW_Fixed = Results
@load "20data_RW_5000Particles2.jld2"
data20_5000Particles_RW_Adaptive = Results

p5 = plotposterior(data20_5000Particles_RW_Adaptive,n,title="Adaptive Stepsize, 5000 Particles")
p6= plotposterior(data20_5000Particles_RW_Fixed,n,title="Fixed Stepsize, 5000 Particles")

plot(p1,p3,p5,p2,p4,p6,layout=(2,3),size=(1800,1200))
