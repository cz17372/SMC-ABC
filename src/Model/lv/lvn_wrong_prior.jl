module lvn_wrong

using Distributions, LinearAlgebra

NoParam = 4
μ = 0.0
σ = 1.0

function ptheta(logθ)
    return sum(logpdf(Normal(μ,σ),logθ))
end

U(u) = sum(logpdf(Normal(0,1),u))

function GenPar()
    return rand(Normal(μ,σ),NoParam)
end

function GenSeed(N)
    return randn(N)
end

function Simulator(N)
    logθ = exp.(GenPar())
    u = randn(N)
    return ϕ(u,logθ)
end

function ConSimulator(N,logθ)
    u = randn(N)
    return ϕ(u,logθ)
end


function ϕ(u,logθ;x0=100.0,y0=100.0,dt=1.0,σx=1.0,σy=1.0)
    θ = exp.(logθ)
    N = length(u) ÷ 2
    xvec = zeros(N+1)
    yvec = zeros(N+1)
    xvec[1] = x0; yvec[1]=y0
    for n = 1:N
        xvec[n+1] = max(xvec[n] + dt*(θ[1]*xvec[n]-θ[2]*xvec[n]*yvec[n])+sqrt(dt)*σx*u[2*n-1],0)
        yvec[n+1] = max(yvec[n] + dt*(θ[4]*xvec[n]*yvec[n]-θ[3]*yvec[n]) + sqrt(dt)*σy*u[2*n],0)
    end
    return [xvec[2:end];yvec[2:end]]
end

function Ψ(u;x0=100.0,y0=100.0,dt=1.0,σx=1.0,σy=1.0)
    logθ = σ*u[1:NoParam] .+ μ
    return ϕ(u[(NoParam+1):end],logθ,x0=x0,y0=y0,dt=dt,σx=σx,σy=σy)
end

function GetPostSample(R)
    Index = findall(R.WEIGHT[:,end] .> 0)
    U = R.U[end][1:NoParam,Index]
    return σ*U .+ μ
end


end
    
