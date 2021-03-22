using Random, LinearAlgebra
using Plots, StatsPlots
using JLD2
theme(:wong2)
include("G-and-K/MCMC.jl")
include("G-and-K/SMC-ABC.jl")

############################ Experiment with 20 Observations ##########################
Random.seed!(123)
zstar = rand(Normal(0,1),20)
θstar = [3.0,1.0,2.0,0.5]
ystar = f.(zstar,θ=θstar)

@load "G-and-K/RWM_COV.jld2"
# Get the ground truth from MCMC 
R_RWM,acc = RWM(100000,RWM_Σ,0.2)


plot(R_RWM[:,3],label="",color=:grey,linewidth=0.5,xtickfontsize=8,dpi=200)
density(R_RWM[50001:end,4],label="",color=:grey,linewidth=3)
sigma = cov(R_RWM[1][20001:end,:])

function get_unique_initials(R,T)
    output = zeros(T)
    for i = 1:T
        output[i] = length(unique(R.DISTANCE[R.ANCESTOR[:,i],i]))
    end
    return output
end

# ------------------------------------

R_RW2       = RWSMCABC(10000,100,20,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Naive2    = NaiveSMCABC(10000,100,20,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Langevin2 = LSMCABC(10000,20,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")


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


density(R_RW2.XI[1,:,90],label="RW-SMC-ABC")
density!(R_RWM[20001:end,1],label="MCMC")