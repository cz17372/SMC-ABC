using Distributions, Random, LinearAlgebra, Plots, StatsPlots,Measures
theme(:ggplot2)

include("src/Sampler/RESMC.jl") # Prangle's Algorithm, "RESMC"
include("src/Sampler/RWSMC.jl") # Novel Latents ABC-SMC
include("src/Sampler/ABCSMC.jl")

include("src/Model/lv/lvn.jl")
include("src/Model/lv/lvn_diffuse_prior.jl")
include("src/Model/lv/lvn_wrong_prior.jl")
include("src/Model/lv/lvu.jl")


Euclidean(x,y) = norm(x .- y)
Random.seed!(1018)
logθ = log.([0.4,0.005,0.05,0.001])
ystar = lvu.ConSimulator(100,logθ)
function plotdata(ystar)
    n = length(ystar) ÷ 2
    plot(ystar[1:n],label="",color=:red,linewidth=2.0,framestyle=:box)
    plot!(ystar[n+1:end],label="",color=:green,linewidth=2.0)
end
plotdata(ystar)
savefig("LV-Observations.pdf")
epsvec = Array{Any,1}(undef,20)
for i = 1:20
    parcand = lvu.GenPar()
    R = RESMC.SMC(5000,parcand,TerminalTol=5.0,model=lvu,Dist=Euclidean,PrintRes = true,y=ystar)
    epsvec[i] = R.EPSILON
end

R = RESMC.SMC(5000,logθ,TerminalTol=5.0,model=lvu,Dist=Euclidean,PrintRes = true,y=ystar)
savefig("RESMC-problem.pdf")


plot(log.(closeepsvec[1]),color=:grey,linewidth=2.0,label="",xlabel="Iteration",ylabel="Log \\epsilon")
for i = 2:20
    plot!(log.(closeepsvec[i]),color=:grey,linewidth=2.0,label="")
end
hline!([log(5.0)],color=:red,linewidth=2.0,linestyle=:dash,label="")
for i = 1:20
    plot!(log.(epsvec[i]),color=:green,linewidth=2.0,label="")
end
current()
plot!(log.(R.EPSILON),label="",linewidth=2.0,color=:blue)





LSMC_Informative = RWSMC.SMC(5000,ystar,lvn,Euclidean,η=0.8,TerminalTol=5.0)
LSMC_Diffuse = RWSMC.SMC(5000,ystar,lvn_diffuse,Euclidean,η=0.8,TerminalTol=5.0)
LSMC_Wrong = RWSMC.SMC(5000,ystar,lvn_wrong,Euclidean,η=0.8,TerminalTol=5.0)
X_Informative = lvn.GetPostSample(LSMC_Informative)
X_Diffuse = lvn_diffuse.GetPostSample(LSMC_Diffuse)
X_Wrong   = lvn_wrong.GetPostSample(LSMC_Wrong)

SMC_Informative = ABCSMC.SMC(5000,ystar,lvn,Euclidean,TerminalTol=5.0,TerminalProb=0.001,η=0.8)
Index = findall(SMC_Informative.WEIGHT[:,end] .> 0)
SMCX_Informative = SMC_Informative.U[end][:,Index]

SMC_Diffuse = ABCSMC.SMC(5000,ystar,lvn_diffuse,Euclidean,TerminalTol=5.0,TerminalProb=0.001,η=0.8)
Index = findall(SMC_Diffuse.WEIGHT[:,end] .> 0)
SMCX_Diffuse = SMC_Diffuse.U[end][:,Index]

SMC_Wrong = ABCSMC.SMC(5000,ystar,lvn_wrong,Euclidean,TerminalTol=5.0,TerminalProb=0.001,η=0.8)
Index = findall(SMC_Wrong.WEIGHT[:,end] .> 0)
SMCX_Wrong = SMC_Wrong.U[end][:,Index]

n=1
p1 = density(X_Informative[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Informative[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=2
p2 = density(X_Informative[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Informative[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=3
p3 = density(X_Informative[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Informative[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=4
p4 = density(X_Informative[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Informative[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

plot(p1,p2,p3,p4,layout=(2,2),size=(800,800))
savefig("LV-Informative-Prior.pdf")

n=1
p1 = density(X_Diffuse[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Diffuse[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=2
p2 = density(X_Diffuse[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Diffuse[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=3
p3 = density(X_Diffuse[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Diffuse[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=4
p4 = density(X_Diffuse[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Diffuse[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

Diffuse = plot(p1,p2,p3,p4,layout=(1,4),size=(1600,400),margin=7.0mm)


n=1
p1 = density(X_Wrong[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Wrong[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=2
p2 = density(X_Wrong[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Wrong[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=3
p3 = density(X_Wrong[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Wrong[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

n=4
p4 = density(X_Wrong[n,:],label="",xlabel="\\theta $(n)",color=:green,linewidth=2.0,size=(400,400))
density!(SMCX_Wrong[n,:],label="",color=:red,linewidth=2.0)
vline!([logθ[n]],color=:grey,linestyle=:dash,label="")

Wrong = plot(p1,p2,p3,p4,layout=(1,4),size=(1600,400),margin=7.0mm)

plot(Diffuse,Wrong,layout=(2,1),size=(1600,800),margin=7.0mm)
savefig("LV-Misspecified-Prior.pdf")