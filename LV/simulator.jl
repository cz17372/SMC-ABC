using Base: sign_mask
using Distributions: delta
using Plots: get_linewidth
using Distributions, Plots, StatsPlots
using Random
using LinearAlgebra
using JLD2
theme(:wong2)
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
@load "DelMoral.jld2"
@load "DelMoral_20data.jld2"
@load "RW_20data.jld2"
@load "RW_100data.jld2"
Random.seed!(17372);
θstar = log.([0.4,0.005,0.05,0.001]);
ustar = rand(Normal(0,1),40);
ystar = ϕ(ustar,θ=θstar);

# Get the plot of posterior density
p1 = get_density(RW_20data,1,length(RW_20data.EPSILON),truepar=θstar[1],color=:green,label="RW-ABC-SMC",xlabel="log(theta_1)")
get_density(DelMoral_20data,1,length(DelMoral_20data.EPSILON),truepar=θstar[1],label="Std-ABC-SMC",color=:black,new=false)
p2 = get_density(RW_20data,2,length(RW_20data.EPSILON),truepar=θstar[2],color=:green,label="RW-ABC-SMC",xlabel="log(theta_2)")
get_density(DelMoral_20data,2,length(DelMoral_20data.EPSILON),truepar=θstar[2],label="Std-ABC-SMC",color=:black,new=false)
p3 = get_density(RW_20data,3,length(RW_20data.EPSILON),truepar=θstar[3],color=:green,label="RW-ABC-SMC",xlabel="log(theta_3)")
get_density(DelMoral_20data,3,length(DelMoral_20data.EPSILON),truepar=θstar[3],label="Std-ABC-SMC",color=:black,new=false)
p4 = get_density(RW_20data,4,length(RW_20data.EPSILON),truepar=θstar[4],color=:green,label="RW-ABC-SMC",xlabel="log(theta_4)")
get_density(DelMoral_20data,4,length(DelMoral_20data.EPSILON),truepar=θstar[4],label="Std-ABC-SMC",color=:black,new=false)
plot(p1,p2,p3,p4,layout=(2,2),size=(1200,1200))
rw_20data_pobs = get_rw_pseudo_obs(RW_20data,length(RW_20data.EPSILON))
std_20data_pobs = get_std_psudo_obs(DelMoral_20data,length(DelMoral_20data.EPSILON))
p1 = plot_data(rw_20data_pobs,linewidth=1.0);plot_obs(ystar,new=false)
p2 = plot_data(std_20data_pobs,linewidth=0.2);plot_obs(ystar,new=false)

plot(p1,p2,layout=(2,1),size=(600,1200))

p1 = plot(RW_20data.AcceptanceProb,xlabel="Iteration",ylabel="Acceptance Probability",label="RW-ABC-SMC",size=(600,600))
plot!(DelMoral_20data.AcceptanceProb,label="Std-ABC-SMC")
p2 = plot(log.(RW_20data.EPSILON),xlabel="Iteration",ylabel="Log-Tolerance",label="RW-ABC-SMC")
plot!(log.(DelMoral_20data.EPSILON),label="Std-ABC-SMC")










R = RandomWalk.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.95,TerminalTol=5.0,TerminalProb=0.01)
R2 = DelMoral.SMC(1000,ystar,InitStep=0.3,MinStep=0.2,MinProb=0.2,IterScheme="Fixed",InitIter=2,PropParMoved=0.99,TolScheme="unique",η=0.95,TerminalTol=0.1,TerminalProb=0.01)
using Plots, StatsPlots
theme(:wong2)


Index = findall(RW_100data.WEIGHT[:,end] .> 0)
X = RW_100data.U[end][:,Index]
Σ = cov(X,dims=2)
Euclidean(x;y) = norm(ϕ(x[5:end],θ=x[1:4]) .- y)
U(x) = sum(logpdf.(Normal(-2,3),x[1:4])) + sum(logpdf.(Normal(0,1),x[5:end]))
function MCMC(N,x0,ϵ;y,δ,Σ)
    oldx = x0
    Ind = 0
    d = length(x0)
    L = cholesky(Σ).L
    for n = 2:N
        newx = oldx .+ δ*L*rand(Normal(0,1),d)
        # newx = rand(MultivariateNormal(oldx,δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newx) - U(oldx)
            if Euclidean(newx,y=y) < ϵ
                oldx = newx
                Ind += 1
            end
        end
    end
    return (oldx,Ind)
end

@time MCMC(10000,X[:,1],1.0,y=ystar,δ = 0.2,Σ = Σ)


function MCMC2(N,x0,ϵ;y,δ,Σ)
    oldx = x0
    Ind = 0
    d = length(x0)
    L = cholesky(Σ).L
    Seed = rand(Normal(0,1),d,N)
    PropMove = δ * L * Seed
    for n = 1:N
        newx = oldx .+ PropMove[:,n]
        # newx = rand(MultivariateNormal(oldx,δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newx) - U(oldx)
            if Euclidean(newx,y=y) < ϵ
                oldx = newx
                Ind += 1
            end
        end
    end
    return (oldx,Ind)
end

@time MCMC2(10000,X[:,1],1.0,y=ystar,δ = 0.2,Σ = Σ)