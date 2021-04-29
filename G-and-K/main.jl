using Random, Distributions, JLD2, LinearAlgebra
# Transformation of standard normal RV's to g-and-k
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

# Set the true static parameters
θ0 = [3.0,1.0,2.0,0.5];
# Generate Artificial Data Sets 
Random.seed!(123);
dat20 = f.(rand(Normal(0,1),20),θ=θ0);
#=
# Random the MCMC sampler
include("G-and-K/MCMC/MCMC.jl")
@load "G-and-K/MCMC/MCMC_COV.jld2"
R_MCMC,α_MCMC = RWM(30000,Σ,0.2,y = dat20,θ0 = rand(Uniform(0,10),4))
plot(R_MCMC[:,4],linewidth=2.0,color=:grey)
density(R_MCMC[15001:end,3],linewidth=2.0,color=:darkgreen,label="")

=#
# Random-walk SMC-ABC
ystar = dat20
include("Langevin/Langevin-SMC-ABC.jl")

include("BPS/BPS-SMC-ABC.jl")

ϵ = 10.0
samp() = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
truemat = zeros(10000,24)
n = 1
while n <= 10000
    x = samp()
    if BPS_SMC_ABC.Dist(x,y=dat20) < ϵ
        println(n)
        truemat[n,:] = x
        n += 1
    end
end
Σ = cov(truemat)
d = mean(mapslices(norm,rand(MultivariateNormal(zeros(24),Σ),100000),dims=1))
mean(mapslices(norm,rand(MultivariateNormal(zeros(24),1.0/d^2*Σ),100000),dims=1))
density(truemat[:,4])

x0 = truemat[2,:]
R = BPS.BPS_LocalMH(50000,x0,0.05,0.9,y=dat20,ϵ=ϵ,Σ = 1.0*I)

u0 = rand(MultivariateNormal(zeros(24),1.0/d^2*Σ))
δ = 0.005
x1,u1 = BPS_SMC_ABC.φ1(x0,u0,δ)
(any([(x1.>10);(x1 .< 0)]))
BPS_SMC_ABC.Dist(x1,y=dat20)
BPS_SMC_ABC.α1(x0,u0,δ,y=dat20,ϵ=ϵ,Σ=1.0/d^2*Σ)

include("BPS/BPS-SMC-ABC.jl")
x0 = truemat[2,:]
X,_ = BPS.BPS_LocalMH(200000,x0,1.0,0.5,y=dat20,ϵ=ϵ,Σ = 1.0/d^2*I)
R = BPS.BPS_SMC_ABC(10000,100,20,y=dat20,Threshold=0.8,δ=0.5,κ=0.6,K0=10)

index = findall(R.DISTANCE[:,end].>0)
density(R.U[5,index,end])