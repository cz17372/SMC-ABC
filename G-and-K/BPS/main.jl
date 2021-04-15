include("G-and-K/BPS/BPS.jl")
using Plots, StatsPlots, LaTeXStrings, Random
theme(:juno)
function groundtruth(N,ϵ)
    u = zeros(N,2)
    n = 1
    while n <= N
        ucand = [rand(Uniform(0,10)),rand(Normal(0,1))]
        if dist(ucand) < ϵ
            u[n,:] = ucand
            n += 1
        end
    end
    return u 
end

Random.seed!(12358)
zstar = rand(Normal(0,1))
ystar = f([0.5,zstar])


R = BPS_SMC_ABC(5000,20,Threshold=0.9,δ=0.1,refresh_rate=0.3,K=20);
R2 = RW_SMC_ABC(5000,20,Threshold = 0.9,δ=0.1,K = 20)
log.(R.EPSILON)
log.(R2.EPSILON)
scatterplot(transpose(R.U[:,findall(R.DISTANCE[:,end] .> 0),end]))
scatterplot(transpose(R2.U[:,findall(R2.DISTANCE[:,end] .> 0),end]))

unique(R.DISTANCE[:,end])
unique(R2.DISTANCE[:,end])


density(R.U[1,findall(R.DISTANCE[:,end] .> 0),end])
density!(R2.U[1,findall(R2.DISTANCE[:,end] .> 0),end])


X = groundtruth(10000,0.1)

scatter(X[:,1],X[:,2],label="",markersize=0.1,markerstrokewidth=0,color=:red)
density(X[:,3])