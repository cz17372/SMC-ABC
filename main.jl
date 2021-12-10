using Distributions, StatsPlots, Plots, KernelDensity, Measures, Random, LinearAlgebra, JLD2
theme(:wong2)

# Define Distance Metrics
##
Euclidean(x,y) = norm(x .- y)
Ordered(x,y)   = norm(sort(x).-sort(y))


##
# Import Samplers
include("src/Sampler/RESMC.jl") # Prangle's Algorithm, "RESMC"
include("src/Sampler/RWSMC.jl") # Novel Latents ABC-SMC
include("src/Sampler/ABCSMC.jl")
include("src/Sampler/MCMC.jl")# Del Moral's ABC-SMC

# g-and-k example
include("src/Model/gk/gku.jl")
include("src/Model/gk/gkn.jl")
function SMC_CompCost(R)
    T = length(R.EPSILON) - 1
    Cost = 0
    for n = 1:T
        Cost += sum(R.WEIGHT[:,n+1] .> 0) * R.K[n]
    end
    return Cost/sum(R.WEIGHT[:,end] .> 0)
end
function gkResults(epsvec::Vector{Float64},M,N,ystar,Dist,θ0)
    k = length(epsvec)
    RWSMC_posterior = Array{Any,2}(undef,k,4)
    ABCSMC_posterior = Array{Any,2}(undef,k,4)
    RESMC_MCMC = Array{Matrix,1}(undef,k)
    RESMC_CompCost = zeros(k)
    RWSMC_CompCost = zeros(k)
    ABCSMC_CompCost = zeros(k)
    for n = 1:k
        # Perform the RWABCSMC
        R = RWSMC.SMC(N,ystar,gkn,Dist,η=0.8,TerminalTol=epsvec[n],TerminalProb=0.01)
        Index = findall(R.WEIGHT[:,end] .> 0)
        X     = gkn.GetPostSample(R)
        for j = 1:4
            RWSMC_posterior[n,j] = kde(X[j,:])
        end
        Σ = cov(X,dims=2)
        RWSMC_CompCost[n] = SMC_CompCost(R)
        R = ABCSMC.SMC(N,ystar,gkn,Dist,η=0.8,TerminalTol=epsvec[n],TerminalProb=0.0001)
        if R.EPSILON[end] == epsvec[n]
            Index = findall(R.WEIGHT[:,end] .> 0)
            X = R.U[end][:,Index]
            for j = 1:4
                ABCSMC_posterior[n,j] = kde(X[j,:])
            end
            ABCSMC_CompCost[n]=SMC_CompCost(R)
        else
            ABCSMC_CompCost[n] = Inf
        end
        R = RESMC.PMMH(θ0,M,N,y=ystar,model=gku,Dist=Dist,Σ=Σ,ϵ=epsvec[n])
        if R == "Infeasible"
            RESMC_CompCost[n] = Inf
        else
            RESMC_MCMC[n] = R.theta
            RESMC_CompCost[n] = sum(R.NumVec)/M
        end
    end
    return (Posterior = (RWABCSMC=RWSMC_posterior,ABCSMC=ABCSMC_posterior,RESMC=RESMC_MCMC),CompCost = (RWABCSMC=RWSMC_CompCost,ABCSMC=ABCSMC_CompCost,RESMC=RESMC_CompCost))
end
function plotres(MCMC,RWSMC,ABCSMC;xlabel,ylabel,layout=(2,2))
    X1 = gkn.GetPostSample(RWSMC)
    X2 = ABCSMC.U[end][:,findall(ABCSMC.WEIGHT[:,end] .> 0)]
    X3 = MCMC.Sample
    picture = Array{Any,1}(undef,4)
    for i = 1:4
        picture[i] = density(X3[50001:end,i],label="",color=:grey,linewidth=2.0,size=(400,400))
        density!(X2[i,:],label="",color=:red,linewidth=2.0)
        density!(X1[i,:],label="",color=:green,linewidth=2.0)
    end
    p = plot(picture[1],picture[2],picture[3],picture[4],layout=layout,xlabel=xlabel,ylabel=ylabel,margin=7.0mm,size=(400*layout[2],400*layout[1]))
    return p
end
function plotmcmc(RESMC,index)
    n = length(index)
    picture = plot(layout=(n,4),size=(400*4,400*n),margin=2.0mm)
    for i = 1:n-1
        plot!(picture,subplot=4*i-3,RESMC[index[i]][:,1],label="",color=:grey,linewitdth=10.0,xticks=:none,grid=:show)
        plot!(picture,subplot=4*i-2,RESMC[index[i]][:,2],label="",color=:grey,linewitdth=10.0,xticks=:none,grid=:show)
        plot!(picture,subplot=4*i-1,RESMC[index[i]][:,3],label="",color=:grey,linewitdth=10.0,xticks=:none,grid=:show)
        plot!(picture,subplot=4*i,RESMC[index[i]][:,4],label="",color=:grey,linewitdth=10.0,xticks=:none,grid=:show)
    end
    plot!(picture,subplot=4*n-3,RESMC[index[n]][:,1],label="",color=:grey,linewitdth=10.0,xticks=0:1000:2000,grid=:show)
    plot!(picture,subplot=4*n-2,RESMC[index[n]][:,2],label="",color=:grey,linewitdth=10.0,xticks=0:1000:2000,grid=:show)
    plot!(picture,subplot=4*n-1,RESMC[index[n]][:,3],label="",color=:grey,linewitdth=10.0,xticks=0:1000:2000,grid=:true)
    plot!(picture,subplot=4*n,RESMC[index[n]][:,4],label="",color=:grey,linewitdth=10.0,xticks=0:1000:2000,grid=:show)
    return picture
end
    
Random.seed!(123)
θstar = [3.0,1.0,2.0,0.5]
ystar20 = gku.ConSimulator(20,θstar)
epsvec = [25.0,20.0,15,10,5,2,1,0.5,0.2]
gk_20data = gkResults([25.0],2000,5000,ystar20,Euclidean,θstar)


# Sample one from MCMC, RWSMC, ABCSMC with ϵ = 0.2, TerminalProb=0.001

ABCSMC20 = ABCSMC.SMC(10000,ystar20,gkn,Euclidean,η=0.8,TerminalTol=0.2,TerminalProb=0.001)
RWSMCX=gkn.GetPostSample(RWSMC20)
ABCSMCX = ABCSMC20.U[end][:,findall(ABCSMC20.WEIGHT[:,end] .> 0)]
Σ = cov(RWSMCX,dims=2)
MCMC20 = MCMC.RWM(100000,Σ,0.2,y=ystar20)
p1 = plotres(MCMC20,RWSMC20,ABCSMC20,xlabel=["a","b","g","k"],ylabel=["Density" "" "Density" ""])
p2 = plotmcmc(gk_twentydata_RESMC.Chains,[6,7,8,9])
plot(p1,p2,layout=(1,2),size=(1400,700))
savefig("gk-20data-res.pdf")



Random.seed!(4013)
θstar = [3.0,1.0,2.0,0.5]
ystar250 = gkn.ConSimulator(250,θstar)
RWSMC250data = RWSMC.SMC(5000,ystar250,gkn,Euclidean,η = 0.8, TerminalTol=0.5)

Random.seed!(4013)
θstar = [3.0,1.0,2.0,0.5]
ystar100 = gkn.ConSimulator(100,θstar)
RWSMC100data = RWSMC.SMC(10000,ystar100,gkn,Euclidean,η = 0.8, TerminalTol=0.5)
@save "temp_gkRWSMC100data.jld2" RWSMC100data

Random.seed!(17372)
θstar = [3.0,1.0,2.0,0.5]
ystar50 = gkn.ConSimulator(50,θstar)
RWSMC50data = RWSMC.SMC(5000,ystar50,gkn,Euclidean,η = 0.8, TerminalTol=0.5)
@save "temp_gkRWSMC50data.jld2" RWSMC50data

@load "temp_gkRWSMC50data.jld2"
@load "temp_gkRWSMC100data.jld2"

dat50 = Array{Any,1}(undef,4)
dat100 = Array{Any,1}(undef,4)
dat250 = Array{Any,1}(undef,4)

for i = 1:4
    dat50[i] = kde(gkn.GetPostSample(RWSMC50data)[i,:])
    dat100[i] = kde(gkn.GetPostSample(RWSMC100data)[i,:])
    dat250[i] = kde(gkn.GetPostSample(RWSMC250data)[i,:])
end

gkRWSMCPosterior = (data50=dat50,data100=dat100,data250=dat250)

MCMC50= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC50data),dims=2),0.3,y=ystar50)
MCMC100= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC100data),dims=2),0.3,y=ystar100)
MCMC250= MCMC.RWM(100000,cov(gkn.GetPostSample(RWSMC250data),dims=2),0.5,y=ystar250)


ABCSMC50 = ABCSMC.SMC(10000,ystar50,gkn,Euclidean,η=0.8,TerminalTol=0.2,TerminalProb=0.001)
ABCSMC100 = ABCSMC.SMC(10000,ystar100,gkn,Euclidean,η=0.8,TerminalTol=0.2,TerminalProb=0.001)

p1 = plotres(MCMC50,RWSMC50data,ABCSMC50,xlabel=["" "" "" ""],ylabel=["Density" "" "" ""],layout=(1,4))
p2 = plotres(MCMC100,RWSMC100data,ABCSMC100,xlabel=["a" "b" "g" "k"],ylabel=["Density" "" "" ""],layout=(1,4))

plot(p1,p2,layout=(2,1),size=(1800,900))

savefig("gk50100data.pdf")

Random.seed!(17372)

θ = rand(Normal(0,2))
y = rand(Normal(θ,5))

sample = zeros(10000,2)
for n =1:10000
    sample[n,1] = rand(Normal(0,2))
    sample[n,2] = rand(Normal(sample[n,1],5))
end

scatter(sample[:,1],sample[:,2],markersize=0.5,markerstrokewidth=0，color=)