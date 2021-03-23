using Random, LinearAlgebra
using Plots, StatsPlots
using JLD2
theme(:wong2)
include("G-and-K/MCMC.jl")
include("G-and-K/SMC-ABC.jl")

############################ Experiment with 100 Observations ##########################
Random.seed!(4013)
zstar = rand(Normal(0,1),100)
θstar = [3.0,1.0,2.0,0.5]
ystar = f.(zstar,θ=θstar)

@load "G-and-K/RWM_COV.jld2"
# Get the ground truth from MCMC 
R_RWM,acc = RWM(100000,RWM_Σ,1.0)
RWM_Σ = cov(R_RWM[50001:end,:])



plot(R_RWM[:,2],label="",color=:grey,linewidth=0.5,xtickfontsize=8,dpi=200)
density(R_RWM[50001:end,4],label="",color=:grey,linewidth=3)
sigma = cov(R_RWM[1][20001:end,:])

function get_unique_initials(R,T)
    output = zeros(T)
    for i = 1:T
        output[i] = length(unique(R.DISTANCE[R.ANCESTOR[:,i],i]))
    end
    return output
end



R_RW2       = RWSMCABC(10000,200,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Naive2    = NaiveSMCABC(10000,200,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Langevin2 = LSMCABC(10000,200,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")

plot(log.(R_Naive2.EPSILON),label="Naive-SMC-ABC",xlabel="Iteration",ylabel="epsilon")
plot!(log.(R_RW2.EPSILON),label="RW-SMC-ABC")
plot!(log.(R_Langevin2.EPSILON),label="L-SMC-ABC")
savefig("epsilon.pdf")

UniqueL = get_unique_initials(R_Langevin2,100)
UniqueRW2 = get_unique_initials(R_RW2,100)
UniqueNaive2 = get_unique_initials(R_Naive2,100)

plot(UniqueNaive2,xlabel="Iteration",label="Naive-SMC-ABC")
plot!(UniqueRW2,label="RW-SMC-ABC")
plot!(UniqueL,label="L-SMC-ABC")
savefig("uniqueparticles.pdf")

n = 1; t= 100
density(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_RWM[20001:end,n],label="MCMC")
density!(R_Naive2.THETA[n,:,t],label="Naive SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")

plot(R_Langevin2.SIGMA)
plot!(R_RW2.SIGMA)

cov(R_Langevin2.XI[:,:,end],dims=2)

function PlotDensity(RW,Langevin,Naive,MCMC,n,t)
    label = ["a","b","g","k"]
    density(MCMC[20001:end,n],label="MCMC",xlabel=label[n],ylabel="density",title="Iteration $t",linewidth=2)
    density!(RW.XI[n,:,t],label="RW-SMC-ABC, epsilon = $(round(RW.EPSILON[t+1];sigdigits=4))",linewidth=2)
    
    density!(Naive.THETA[n,:,t],label="Naive SMC-ABC,epsilon = $(round(Naive.EPSILON[t+1];sigdigits=4))",linewidth=2)
    density!(Langevin.XI[n,:,t],label="L-SMC-ABC,epsilon = $(round(Langevin.EPSILON[t+1];sigdigits=4))",linewidth=2)
end

plot(log.(R_Naive2.EPSILON),label="Naive-SMC-ABC",xlabel="Iteration",ylabel="epsilon",size=(1400,1400),framestyle=:box,legendfontsize=11,linewidth=2)
plot!(log.(R_RW2.EPSILON),label="RW-SMC-ABC",linewidth=2)
plot!(log.(R_Langevin2.EPSILON),label="L-SMC-ABC",linewidth=2)
savefig("epsilon.pdf")

t = 60
p1 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,1,t);
p2 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,2,t);
p3 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,3,t);
p4 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,4,t);
plot(p1,p2,p3,p4,layout=(2,2),size=(1400,1400),framestyle=:box,legendfontsize=11)
savefig("iteration60.pdf")

t = 80
p1 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,1,t);
p2 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,2,t);
p3 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,3,t);
p4 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,4,t);
plot(p1,p2,p3,p4,layout=(2,2),size=(1400,1400),framestyle=:box,legendfontsize=11)
savefig("iteration80.pdf")

t = 100
p1 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,1,t);
p2 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,2,t);
p3 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,3,t);
p4 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,4,t);
plot(p1,p2,p3,p4,layout=(2,2),size=(1400,1400),framestyle=:box,legendfontsize=11)
savefig("iteration100.pdf")


t = 120
p1 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,1,t);
p2 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,2,t);
p3 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,3,t);
p4 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,4,t);
plot(p1,p2,p3,p4,layout=(2,2),size=(1400,1400),framestyle=:box,legendfontsize=11)
savefig("iteration120.pdf")

t = 150
p1 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,1,t);
p2 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,2,t);
p3 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,3,t);
p4 = PlotDensity(R_RW2,R_Langevin2,R_Naive2,R_RWM,4,t);
plot(p1,p2,p3,p4,layout=(2,2),size=(1400,1400),framestyle=:box,legendfontsize=11)
savefig("iteration150.pdf")