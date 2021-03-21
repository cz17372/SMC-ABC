
using Random, LinearAlgebra
using Plots, StatsPlots
using JLD2
theme(:mute)
include("G-and-K/MCMC.jl")
include("G-and-K/SMC-ABC.jl")
Random.seed!(17372)
zstar = rand(Normal(0,1),100)
θstar = [3.0,1.0,2.0,0.5]
ystar = f.(zstar,θ=θstar)

@load "G-and-K/data.jld2"; ystar = y0;
@load "G-and-K/RWM_COV.jld2"

R_RWM,acc = RWM(100000,RWM_Σ,0.2)


plot(R_RWM[1][:,4])
density(R_RWM[1][50001:end,4])
sigma = cov(R_RWM[1][20001:end,:])


R_Langevin = LSMCABC(10000,100,100,Threshold=0.95,σ=0.3,λ=1.0) # time taken = 31mins
R_Naive    = NaiveSMCABC(10000,100,20,Threshold=0.95,σ=0.3,λ=1.0) # time taken = 18 seconds
R_RW       = RWSMCABC(10000,100,20,Threshold=0.95,σ=0.3,λ=1.0) # 1min 18s


plot(log.(R_Naive.EPSILON),label="Naive-SMC-ABC",xlabel="Iteration",ylabel="epsilon")
plot!(log.(R_RW.EPSILON),label="RW-SMC-ABC")
plot!(log.(R_Langevin.EPSILON),label="L-SMC-ABC")

function get_unique_initials(R,T)
    output = zeros(T)
    for i = 1:T
        output[i] = length(unique(R.DISTANCE[R.ANCESTOR[:,i],i]))
    end
    return output
end


UniqueL = get_unique_initials(R_Langevin,300)
UniqueRW = get_unique_initials(R_RW,300)
UniqueNaive = get_unique_initials(R_Naive,300)

plot(UniqueNaive,label="Naive-SMC-ABC",xlabel="Iteration",ylabel="No. Unique Initials")
plot!(UniqueRW,label="RW-SMC-ABC")
plot!(UniqueL,label="L-SMC-ABC")

n = 1; t = 200
density(R_RWM[20001:end,n],label="RW-MH")
density!(R_RW.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive.THETA[n,:,t])

# ------------------------------------

R_RW2       = RWSMCABC(10000,100,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Naive2    = NaiveSMCABC(10000,100,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Langevin2 = LSMCABC(10000,100,100,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")


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

n = 1; t = 300
p1=density(R_RWM[20001:end,n],label="RW-MH",xlabel="a",ylabel="Posterior")
density!(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive2.THETA[n,:,t],label="Naive-SMC-ABC")
n=2
p2=density(R_RWM[20001:end,n],label="RW-MH",xlabel="b",ylabel="Posterior")
density!(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive2.THETA[n,:,t],label="Naive-SMC-ABC")
n=3
p3=density(R_RWM[20001:end,n],label="RW-MH",xlabel="g",ylabel="Posterior")
density!(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive2.THETA[n,:,t],label="Naive-SMC-ABC")
n=4
p4=density(R_RWM[20001:end,n],label="RW-MH",xlabel="K",ylabel="Posterior")
density!(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive2.THETA[n,:,t],label="Naive-SMC-ABC")
plot(p1,p2,p3,p4,layout=(2,2),size=(1000,800))
savefig("density.pdf")





density(R_Naive2.THETA[4,:,100])
density!(R_RW2.XI[4,:,end])
density!(R_Langevin2.XI[4,:,end])
plot(log.(R_Naive2.EPSILON))
plot!(log.(R_RW2.EPSILON))
plot!(log.(R_Langevin2.EPSILON))