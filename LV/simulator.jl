using Distributions, Plots
using ForwardDiff:gradient
using Random
using LinearAlgebra
function Simulate(params)
    r0 = 100.0
    f0 = 100.0
    N  = 250
    δt = 0.2
    σr = 1.0
    σf = 1.0
    rvec = zeros(N+1)
    fvec = zeros(N+1)
    rvec[1] = r0
    fvec[1] = f0
    for n = 2:(N+1)
        rvec[n] = rvec[n-1] + δt*(params[1]*rvec[n-1] - params[2]*rvec[n-1]*fvec[n-1]) + sqrt(δt)*σr*rand(Normal(0,1))
        if rvec[n] < 0
            rvec[n] = 0.0
        end
        fvec[n] = fvec[n-1] + δt*(params[4]*rvec[n-1]*fvec[n-1]-params[3]*fvec[n-1]) + sqrt(δt)*σf*rand(Normal(0,1))
        if fvec[n] < 0
            fvec[n] =  0.0
        end
    end
    return rvec,fvec
end
Random.seed!(123)
params = [0.4,0.005,0.05,0.001]
obs = Simulate(params)
plot(obs[1])
plot(r,label="prey"); plot!(f,label="predator")

r = zeros(1000,251)
f = zeros(1000,251)
for i = 1:1000
    r[i,:],f[i,:] = Simulate(params)
end
plot(r[1,:],color=:green,linewidth=0.1,label="prey");
for i = 2:1000
    plot!(r[i,:],label="",color=:green,linewidth=0.1);
end
plot!(f[1,:],color=:red,linewidth=0.1,label="predator");
for i = 2:1000
    plot!(f[i,:],label="",color=:red,linewidth=0.1);
end
current()

function ϕ(x)
    N = (length(x) - 4) ÷ 2
    params = x[1:4]
    r0 = 100.0
    f0 = 100.0
    normvecr = x[5:(4+N)]
    normvecf = x[(5+N):end]
    δt = 0.2
    σr = 1.0
    σf = 1.0
    rvec = zeros(N+1)
    fvec = zeros(N+1)
    rvec[1] = r0
    fvec[1] = f0
    for n = 2:(N+1)
        rvec[n] = rvec[n-1] + δt*(params[1]*rvec[n-1] - params[2]*rvec[n-1]*fvec[n-1]) + sqrt(δt)*σr*normvecr[n-1]
        if rvec[n] < 0
            rvec[n] = 0.0
        end

        fvec[n] = fvec[n-1] + δt*(params[4]*rvec[n-1]*fvec[n-1]-params[3]*fvec[n-1]) + sqrt(δt)*σf*normvecf[n-1]
        if fvec[n] < 0
            fvec[n] =  0.0
        end
    end
    return rvec,fvec
end

x = [params;rand(Normal(0,1),500)]
y = ϕ(x)
function dist(x;obs)
    pyseudo_obs = ϕ(x)
    return norm([(pyseudo_obs[1] .- obs[1]);(pyseudo_obs[2] .- obs[2])])
end


using Flux
