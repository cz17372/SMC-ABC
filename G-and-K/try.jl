
include("G-and-K/MCMC.jl")
@load "G-and-K/RWM_COV.jld2" 
R_RWM = RWM(100000,RWM_Σ,0.2)
using Plots, StatsPlots

R_Langevin = LSMCABC(10000,300,20,Threshold=0.95,σ=0.3,λ=1.0) # time taken = 31mins
R_Naive    = NaiveSMCABC(10000,300,20,Threshold=0.95,σ=0.3,λ=1.0) # time taken = 18 seconds
R_RW       = RWSMCABC(10000,300,20,Threshold=0.95,σ=0.3,λ=1.0) # 1min 18s


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

n = 4; t = 300
density(R_RWM.Sample[20001:end,n],label="RW-MH")
density!(R_RW.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive.THETA[n,:,t])

# ------------------------------------

R_RW2       = RWSMCABC(10000,100,20,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Naive2    = NaiveSMCABC(10000,100,20,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")
R_Langevin2 = LSMCABC(10000,100,20,Threshold=0.9,σ=0.3,λ=1.0,Method="Unique")


plot(log.(R_Naive2.EPSILON),label="Naive-SMC-ABC",xlabel="Iteration",ylabel="epsilon")
plot!(log.(R_RW2.EPSILON),label="RW-SMC-ABC")
plot!(log.(R_Langevin2.EPSILON),label="L-SMC-ABC")

UniqueL = get_unique_initials(R_Langevin2,100)
UniqueRW2 = get_unique_initials(R_RW2,100)
UniqueNaive2 = get_unique_initials(R_Naive2,100)

plot(UniqueNaive2);plot!(UniqueRW2);plot!(UniqueL)
plot(R_Langevin2.SIGMA)

n = 1; t = 50
density(R_RWM.Sample[20001:end,n],label="RW-MH")
density!(R_RW2.XI[n,:,t],label="RW-SMC-ABC")
density!(R_Langevin2.XI[n,:,t],label="L-SMC-ABC")
density!(R_Naive2.THETA[n,:,t],label="Naive-SMC-ABC")

histogram(R_RW2.XI[4,:,250],bins=200,color=:grey,normalize=true)
density!(R_RWM.Sample[20001:end,4],label="RW-MH")