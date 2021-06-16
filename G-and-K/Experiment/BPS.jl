# This is an experiment on 20 data with RW-SMC-ABC method

using Distributions, Random, JLD2
println("Enter the seeds:")
seed = readline()
seed = parse(Int64,seed)
println("Enter the number of data:")
nodata = readline()
nodata = parse(Int64,nodata)
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
θ0 = [3.0,1.0,2.0,0.5];
Random.seed!(seed);
dat20 = f.(rand(Normal(0,1),nodata),θ=θ0);

include("../BPS/BPS-SMC-ABC.jl")
println("Your are using BPS SMC-ABC method now")
println("Enter the number of particles for each iteration:")
N = readline()
N = parse(Int64,N)
println("Enter the number of SMC-ABC steps performed:")
T = readline()
T = parse(Int64,T)


println("Eneter the Threshold value:")
Threshold = readline()
Threshold = parse(Float64,Threshold)




println("Initial Stepsize:")
δ = readline()
δ = parse(Float64,δ)

println("Refreshing Rate:")
κ = readline()
κ = parse(Float64,κ)

println("Maximum number of bounces proposed outside the boundary:")
MaxBounce = readline()
MaxBounce = parse(Int64,MaxBounce)

println("The minimum acceptance probability required:")
AcceptLowerbound = readline()
AcceptLowerbound = parse(Float64,AcceptLowerbound)

println("Minimum Step Size:")
MinStepSize = readline()
MinStepSize = parse(Float64,MinStepSize)


println("No of simulations performed")
m = readline()
m = parse(Int64,m)


Information = (seed=seed,NoData=nodata,N=N,T=T,Threshold=Threshold,κ=κ,MaxBounce=MaxBounces,MinAcceptProb=AcceptLowerbound,MaxStep=δ,MinStep=MinStepSize)

EPSILON = Array{Any,1}(undef,m)
K       = Array{Any,1}(undef,m)
α       = Array{Any,1}(undef,m)
Theta   = Array{Any,1}(undef,m)


for i = 1:m
    R = BPS.SMC(N,T,dat20,Threshold=Threshold,δ = δ,κ=κ,K0 = 5, MH=BPS.BPS1,Dist=BPS.Dist2,MaxBounce=MaxBounce,AcceptLowerbound=AcceptLowerbound,MinStepSize=MinStepSize)
    EPSILON[i] = R.EPSILON
    K[i]       = R.K 
    α[i]       = R.AcceptanceProb
    Index      = findall(R.WEIGHT[:,end] .> 0)
    Theta[i]   = R.U[1:4,Index,end]
end

Results = (Information=Information, EPSILON=EPSILON,K=K,alpha=α,Theta=Theta)

println("The name for the result file:")
name = readline()
@save name Results