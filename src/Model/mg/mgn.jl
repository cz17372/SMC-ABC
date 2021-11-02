module mgn

using Distributions, LinearAlgebra

function ptheta(θ)
    return logpdf(Uniform(0,1/3),θ[1])+logpdf(Uniform(0,10),θ[2])+logpdf(Uniform(θ[2],θ[2]+10),θ[3])
end

U(u) = sum(logpdf(Normal(0,1),u))


function GenSeed(N)
    return randn(N)
end

function ϕ(u,θ)
    N = length(u)÷2
    avec = zeros(N)
    svec = zeros(N)
    for i = 1:N
        avec[i] = -1/θ[1]*log(cdf(Normal(0,1),u[i]))
        svec[i] = θ[2] + (θ[3]-θ[2])*cdf(Normal(0,1),u[i+N])
    end
    Avec = zeros(N); Dvec=zeros(N)
    Avec = cumsum(avec)
    dvec = zeros(N); dvec[1] = svec[1] + max(0,Avec[1]); Dvec[1] = dvec[1]
    for n = 2:N
        dvec[n] = svec[n] + max(0,Avec[n]-Dvec[n-1])
        Dvec[n] = sum(dvec[1:n])
    end
    return [Avec;Dvec]
end

function Ψ(u)
    θ1 = cdf(Normal(0,1),u[1])/3
    θ2 = 10*cdf(Normal(0,1),u[2])
    θ3 = θ2 + 10*cdf(Normal(0,1),u[3])
    θ = [θ1,θ2,θ3]
    return ϕ(u[4:end],θ)
end


end
