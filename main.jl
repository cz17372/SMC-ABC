using Distributions, Random, LinearAlgebra, Plots, StatsPlots
theme(:ggplot2)
include("src/RW.jl")
include("src/RESMC.jl")
include("src/gkn.jl")
include("src/gku.jl")

θstar = [3.0,1.0,2.0,0.5]
Random.seed!(123)
ystar = gkn.ConSimulator(250,θstar)

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
<<<<<<< HEAD
R = RW.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=10.0,η = 0.8,gc=false,TerminalProb=0.0,MinStep=0.1)
utils.RWSMC_CompCost(R)


Index = findall(R.WEIGHT[:,end].>0); U = R.U[end][1:4,Index]; X = 10*cdf(Normal(0,1),U)
Σ = cov(X,dims=2)

R2 = RESMC.PMMH(θstar,2000,10000,y=ystar,model=gku,Dist=Dist,ϵ=10.0,Σ=Σ)
=======
R = RW.SMC(10000,ystar,length(ystar)+4,gkn,SumDist,TerminalTol=0.5,η = 0.8,gc=true,TerminalProb=0.0,MinStep=0.1)

Index = findall(R.WEIGHT[:,end] .> 0)
U = R.U[end][:,Index]
X = 10*cdf(Normal(0,1),U[1:4,:])

density(X[1,:],size=(600,600),linewidth=2,color=:darkolivegreen,xlim=(2,4))
ave = zeros(24,length(R.U))
for n = 1:size(ave)[2]
    Index = findall(R.WEIGHT[:,n] .> 0)
    U = R.U[n][:,Index]
    ave[:,n] = mapslices(mean,U,dims=2)
end


plot(ave[4,:])
>>>>>>> a6b3d1b66efbc3c01f586f1d229615da24744685

plot(R2.theta[:,1])