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


u0 = U[:,1]

Dist(gkn.ϕ(u0),ystar)

u1 = [rand(MultivariateNormal(u0[1:4],0.1*Cov1));u0[5:end]]
u2 = rand(MultivariateNormal(u0,0.1*cov(U,dims=2)))

Dist(gkn.ϕ(u2),ystar)

ind = 0
dist = 0.0
for i = 1:size(U)[2]
    u0 = U[:,i]
    u1 = [rand(MultivariateNormal(u0[1:4],0.1*Cov1));rand(MultivariateNormal(u0[5:end],0.1*Cov2))]
    dist += norm(u1 .- u0)
    if Dist(gkn.ϕ(u1),ystar) < 0.5
        ind += 1
    end
end

ind/size(U)[2]
dist/size(U)[2]

ind = 0
dist = 0.0
for i = 1:size(U)[2]
    u0 = U[:,i]
    u1 = rand(MultivariateNormal(u0,0.1*cov(U,dims=2)))
    dist += norm(u1 .- u0)
    if Dist(gkn.ϕ(u1),ystar) < 0.5
        ind += 1
    end
end

