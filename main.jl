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
R = RW.SMC(5000,ystar,length(ystar)+4,gkn,Dist,TerminalTol=10.0,η = 0.8,gc=false,TerminalProb=0.0,MinStep=0.1)
utils.RWSMC_CompCost(R)

