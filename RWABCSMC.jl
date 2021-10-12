using Distributions, Random, LinearAlgebra
using Plots, StatsPlots
using KernelDensity
include("src/gku.jl")
include("src/gkn.jl")
include("src/RWSMC.jl")
include("src/utils.jl")
include("src/ABCSMC.jl")
include("src/RESMC.jl")
include("src/MCMC.jl")

θstar = [3.0,1.0,2.0,0.5]
Random.seed!(123)
ystar = gkn.ConSimulator(20,θstar)

Dist(x,y) = norm(x .- y)
function SumDist(x,y)
    Ex = quantile(x,collect(1/8:1/8:1))
    Ey = quantile(y,collect(1/8:1/8:1))
    Sx = zeros(4)
    Sy = zeros(4)
    Sx[1] = Ex[4]; Sy[1] = Ey[4]
    Sx[2] = Ex[6] - Ex[2]; Sy[2] = Ey[6] - Ey[2]
    Sx[3] = (Ex[6] + Ex[2] - 2*Ex[4])/Sx[2]
    Sy[3] = (Ey[6] + Ey[2] - 2*Ey[4])/Sy[2]
    Sx[4] = (Ex[7]-Ex[5]+Ex[3]-Ex[1])/Sx[2]
    Sy[4] = (Ey[7]-Ey[5]+Ey[3]-Ey[1])/Sy[2]
    return norm(Sx .- Sy)
end
Dist2(x,y) = norm(sort(x) .- sort(y))


Tolerances = [25,20,15,10,5,2,1,0.5]
CompCostVec20SMC = zeros(8)
CompCostVec20RWSMC = zeros(8)
CompCostVec20RESMC = zeros(8)
for i = 1:8
    R = RWSMC.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=Tolerances[i],η = 0.8,gc=false,TerminalProb=0.015,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec20RWSMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec20RWSMC[i] = Inf
    end
    Index = findall(R.WEIGHT[:,end].>0); Σ = cov(10*cdf(Normal(0,1),R.U[end][1:4,Index]),dims=2)
    R = ABCSMC.SMC(5000,ystar,gkn,Dist,TerminalTol=Tolerances[i],TerminalProb=0.0001,η=0.8,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec20SMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec20SMC[i] = Inf
    end
    R = RESMC.PMMH(θstar,2000,5000,y=ystar,model=gku,Dist=Dist,ϵ=Tolerances[i],Σ=Σ)
    CompCostVec20RESMC[i] = sum(R.NumVec)/2000
end
RWSMC20 = RWSMC.SMC(10000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=5.0,η = 0.8,gc=true,TerminalProb=0.01,MinStep=0.1)
SMC20 = ABCSMC.SMC(10000,ystar,gkn,Dist,TerminalTol=0.5,TerminalProb=0.01,η=0.8,MinStep=0.1)
Index = findall(RWSMC20.WEIGHT[:,end] .> 0)
X = 10*cdf(Normal(0,1),RWSMC20.U[end][1:4,Index])
Σ = cov(X,dims=2)
RESMC20 = RESMC.PMMH(θstar,2000,5000,y=ystar,model=gku,Dist=Dist,ϵ=5.0,Σ=Σ,η=0.8)
plot(RESMC20.theta[:,1])
MCMC20,alpha  = MCMC.RWM(100000,Σ,0.2,y=ystar)
plot(MCMC20[:,4])
theme(:ggplot2)
groundtruth20_a=density(MCMC20[50001:end,1],label="",color=:grey,linewidth=2.0)
groundtruth20_b=density(MCMC20[50001:end,2],label="",color=:grey,linewidth=2.0)
groundtruth20_g=density(MCMC20[50001:end,3],label="",color=:grey,linewidth=2.0)
groundtruth20_k=density(MCMC20[50001:end,4],label="",color=:grey,linewidth=2.0)
RWSMC20_05_a = kde(utils.gkn_getsamp(RWSMC20)[1,:])
RWSMC20_05_b = kde(utils.gkn_getsamp(RWSMC20)[2,:])
RWSMC20_05_g = kde(utils.gkn_getsamp(RWSMC20)[3,:])
RWSMC20_05_k = kde(utils.gkn_getsamp(RWSMC20)[4,:])
plot!(groundtruth20_a,RWSMC20_05_a,label="RW-ABC-SMC",color=:green,width=2)








Random.seed!(17372)
ystar = gkn.ConSimulator(50,θstar)
CompCostVec50 = zeros(8)
CompCostVec50SMC = zeros(8)
CompCostVec50RWSMC = zeros(8)
for i = 1:8
    R = RWSMC.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=Tolerances[i],η = 0.8,gc=false,TerminalProb=0.015,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec50RWSMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec50RWSMC[i] = Inf
    end
    R = ABCSMC.SMC(5000,ystar,gkn,Dist,TerminalTol=Tolerances[i],TerminalProb=0.01,η=0.8,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec50SMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec50SMC[i] = Inf
    end
end

RWSMC50 = RWSMC.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=0.5,η = 0.8,gc=true,TerminalProb=0.015,MinStep=0.1)
SMC50 = ABCSMC.SMC(5000,ystar,gkn,Dist,TerminalTol=0.5,TerminalProb=0.01,η=0.8)
Index = findall(RWSMC50.WEIGHT[:,end] .> 0)
X = 10*cdf(Normal(0,1),RWSMC50.U[end][1:4,Index])
Σ = cov(X,dims=2)
MCMC50,alpha  = MCMC.RWM(100000,Σ,0.2,y=ystar)


Random.seed!(4013)
ystar = gkn.ConSimulator(100,θstar)
CompCostVec100SMC = zeros(8)
CompCostVec100RWSMC = zeros(8)
for i = 1:8
    R = RWSMC.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=Tolerances[i],η = 0.8,gc=false,TerminalProb=0.015,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec100RWSMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec100RWSMC[i] = Inf
    end
    R = ABCSMC.SMC(5000,ystar,gkn,Dist,TerminalTol=Tolerances[i],TerminalProb=0.01,η=0.8,MinStep=0.1)
    if R.EPSILON[end] == Tolerances[i]
        CompCostVec100SMC[i] = utils.RWSMC_CompCost(R)
    else 
        CompCostVec100SMC[i] = Inf
    end
end

RWSMC100 = RWSMC.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=0.5,η = 0.9,gc=true,TerminalProb=0.015,MinStep=0.1)
SMC100 = ABCSMC.SMC(10000,ystar,gkn,Dist,TerminalTol=0.5,TerminalProb=0.01)
Index = findall(RWSMC100.WEIGHT[:,end] .> 0)
X = 10*cdf(Normal(0,1),RWSMC100.U[end][1:4,Index])
Σ = cov(X,dims=2)
MCMC100,alpha  = MCMC.RWM(100000,Σ,0.5,y=ystar)



function plotres(RWSMC,SMC,MCMC)
    Index = findall(RWSMC.WEIGHT[:,end].>0)
    RWSMCX = 10*cdf(Normal(0,1),RWSMC.U[end][1:4,Index])
    Index = findall(SMC.WEIGHT[:,end] .> 0)
    SMCX  = SMC.U[end][:,Index]
    RWSMCepsilon = round(RWSMC.EPSILON[end],digits=3)
    SMCepsilon = round(SMC.EPSILON[end],digits=3)
    p1 = density(MCMC[50001:end,1],label="MCMC",color=:grey,linewidth=2.0,xlabel="a",ylabel="Density",size=(600,600))
    density!(RWSMCX[1,:],label="RW-ABC-SMC,epsilon=$(RWSMCepsilon)",color=:green,linewidth=2)
    density!(SMCX[1,:],label="ABC-SMC,epsilon=$(SMCepsilon)",color=:blue,linewidth=2)
    vline!([3.0],label="",color=:red,linewidth=2.0)
    p2 = density(MCMC[50001:end,2],label="MCMC",color=:grey,linewidth=2.0,xlabel="b",ylabel="Density")
    density!(RWSMCX[2,:],label="RW-ABC-SMC,epsilon=$(RWSMCepsilon)",color=:green,linewidth=2)
    density!(SMCX[2,:],label="ABC-SMC,epsilon=$(SMCepsilon)",color=:blue,linewidth=2)
    vline!([1.0],label="",color=:red,linewidth=2.0)
    p3 = density(MCMC[50001:end,3],label="MCMC",color=:grey,linewidth=2.0,xlabel="g",ylabel="Density")
    density!(RWSMCX[3,:],label="RW-ABC-SMC,epsilon=$(RWSMCepsilon)",color=:green,linewidth=2)
    density!(SMCX[3,:],label="ABC-SMC,epsilon=$(SMCepsilon)",color=:blue,linewidth=2)
    vline!([2.0],label="",color=:red,linewidth=2.0)
    p4 = density(MCMC[50001:end,4],label="MCMC",color=:grey,linewidth=2.0,xlabel="k",ylabel="Density")
    density!(RWSMCX[4,:],label="RW-ABC-SMC,epsilon=$(RWSMCepsilon)",color=:green,linewidth=2)
    density!(SMCX[4,:],label="ABC-SMC,epsilon=$(SMCepsilon)",color=:blue,linewidth=2)
    vline!([0.5],label="",color=:red,linewidth=2.0)
    return [p1,p2,p3,p4]
end

plotres(RWSMC50,SMC50,MCMC50)
savefig("50datares.pdf")

scatter(RWSMC20.K,label="20 data",color=:grey,markersize=2,xlabel="Iteration",ylabel="Average number of MCMC steps",markerstrokewidth=0)
scatter!(RWSMC50.K,label="50,data",color=:green,markersize=2,markerstrokewidth=0)
scatter!(RWSMC100.K,label="100 data",color=:blue,markersize=2,markerstrokewidth=0)
savefig("MCMCsteps.pdf")