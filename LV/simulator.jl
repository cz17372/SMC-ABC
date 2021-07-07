using Plots: Random
using Distributions, Plots, StatsPlots
using Random
using LinearAlgebra
cd("LV")
include("DelMoralABCSMC.jl")
include("RandomWalk.jl")
function ϕ(u;θ)
    θ = exp.(θ)
    N = length(u) ÷ 2
    r0 = 100.0; f0 = 100.0; dt = 1.0; σr = 1.0; σf = 1.0; 
    rvec = zeros(N+1)
    fvec = zeros(N+1)
    rvec[1] = r0; fvec[1] = f0
    for n = 1:N
        rvec[n+1] = max(rvec[n] + dt*(θ[1]*rvec[n]-θ[2]*rvec[n]*fvec[n]) + sqrt(dt)*σr*u[2*n-1],0)
        fvec[n+1] = max(fvec[n] + dt*(θ[4]*rvec[n]*fvec[n]-θ[3]*fvec[n]) + sqrt(dt)*σf*u[2*n],0)
    end
    return [rvec[2:end];fvec[2:end]]
end
function SimulateOne(θ,N)
    u = rand(Normal(0,1),2*N)
    return ϕ(u,θ=θ)
end
Random.seed!(17372);
θstar = log.([0.4,0.005,0.05,0.001]);
ustar = rand(Normal(0,1),100);
ystar = ϕ(ustar,θ=θstar);


R = RandomWalk.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01)

R3 = RandomWalk.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01)
R2 = DelMoral.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=0.1,TerminalProb=0.01)
Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
Index2 = findall(R2.WEIGHT[:,end] .> 0)
X2 = R2.U[end][:,Index2]
using Plots, StatsPlots
p1 = density(X[1,:],label="RW-ABC-SMC",size=(600,600),xlabel="log(theta_1)"); vline!([log(0.4)],label="");density!(X2[1,:],label="Standard ABC-SMC")
p2 = density(X[2,:],label="RW-ABC-SMC",size=(600,600),xlabel="log(theta_2)"); vline!([log(0.005)],label="");density!(X2[2,:],label="Standard ABC-SMC")
p3 = density(X[3,:],label="RW-ABC-SMC",size=(600,600),xlabel="log(theta_3)");vline!([log(0.05)],label="");density!(X2[3,:],label="Standard ABC-SMC")
p4 = density(X[4,:],label="RW-ABC-SMC",size=(600,600),xlabel="log(theta_4)");vline!([log(0.001)],label="");density!(X2[4,:],label="ABC-SMC")
 
plot(R.AcceptanceProb,label="RW-ABC-SMC")
plot!(R2.AcceptanceProb,label="Standard ABC-SMC")


plot(p1,p2,p3,p4,layout=(2,2),size=(1200,1200))

plot(R2.X[end][1:20,Index2[1]],label="",linewidth=0.02,color=:grey);
for i = 2:length(Index2)
    plot!(R2.X[end][1:20,Index2[i]],label="",linewidth=0.02,color=:grey);
end
plot!(ystar[1:20],color=:red,linewidth=3.0,label="");

plot!(R2.X[end][21:40,Index2[1]],label="",linewidth=0.02,color=:grey);
for i = 2:length(Index2)
    plot!(R2.X[end][21:40,Index2[i]],label="",linewidth=0.02,color=:grey);
end
plot!(ystar[21:40],color=:green,linewidth=3.0,label="")

plot(log.(R.EPSILON))
plot!(log.(R2.EPSILON))

g(x) = ϕ(x[5:end],θ=x[1:4])
plot(R.time ./ R.K)
plot!(R2.time ./ R2.K)
x0 = X[:,1]

RandomWalk.Euclidean(x0,y=ystar)
R = @time RandomWalk.MCMC(10000,x0,1.0,y=ystar,δ=0.2,Σ = Σ);
Euclidean(x;y) = norm(ϕ(x[5:end],θ=x[1:4]) .- y)
U(x) = sum(logpdf.(Normal(-2,3),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
function MCMC(N,x0,ϵ;y,δ,Σ)
    oldx = x0
    Ind = 0
    d = length(x0)
    @timeit to "Factorization" L = cholesky(Σ).L
    for n = 2:N
        @timeit to "Sampling" newx = oldx .+ δ*L*rand(Normal(0,1),d)
        if log(rand(Uniform(0,1))) < @timeit to "Prior" U(newx) - U(oldx)
            if @timeit to  "Distance" Euclidean(newx,y=y) < ϵ
                oldx = newx
                Ind += 1
            end
        end
    end
    return (oldx,Ind)
end

to = TimerOutput()
MCMC(10000,x0,1.0,y=ystar,δ=0.2,Σ = Σ)
show(to)
C = cholesky(Σ)
C.L == transpose(C.U)

plot(R.AcceptanceProb)
plot!(R2.AcceptanceProb)