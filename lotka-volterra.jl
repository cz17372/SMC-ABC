using Distributions, Random, LinearAlgebra


include("src/Sampler/RESMC.jl") # Prangle's Algorithm, "RESMC"
include("src/Sampler/RWSMC.jl") # Novel Latents ABC-SMC
include("src/Sampler/ABCSMC.jl")


include("src/Model/lv/lvn.jl")
include("src/Model/lv/lvu.jl")

Random.seed!(1018)
logθ = log.([0.4,0.005,0.05,0.001])
ystar = lvu.ConSimulator(100,logθ)

R = RWSMC.SMC(5000,ystar,lvn,Euclidean,η=0.8,TerminalTol=2.0)
X = lvn.GetPostSample(R)
R = ABCSMC.SMC(5000,ystar,gkn,Euclidean,η=0.8,TerminalTol=5.0,TerminalProb=0.001)

