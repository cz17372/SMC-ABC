x0 = rand(Uniform(0,1),24)
dist(x0)

@btime RWMH(100,x0,270,sigma,0.1)

ProfileView.@profview RWMH(10000,x0,270,sigma,0.1)

@time R = RWMH(10000,x0,270,sigma,0.1)

L = cholesky(sigma).L

a = rand(Normal(0,1),24,10000)

b = L * a
cov(b,dims=2)


function tRWMH(N,x0,ϵ,Σ,δ)
    D = length(x0)
    X = zeros(N,length(x0))
    L = cholesky(Σ).L
    X[1,:] = x0
    for n = 2:N
        xcand = X[n-1,:] .+ δ *L * rand(Normal(0,1),D)
        α = min(0,logpi(xcand,ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
        else
            X[n,:] = X[n-1,:]
        end
    end
    return X[end,:]
end

function t2RWMH(N,x0,ϵ,Σ,δ)
    D = length(x0)
    X = zeros(N,length(x0))
    L = cholesky(Σ).L
    X[1,:] = x0
    for n = 2:N
        X[n,:] = X[n-1,:] .+ δ *L * rand(Normal(0,1),D)
        α = min(0,logpi(X[n,:],ϵ=ϵ)-logpi(X[n-1,:],ϵ=ϵ))
        if log(rand(Uniform(0,1))) > α
            X[n,:] = X[n-1,:]
        end
    end
    return X[end,:]
end

@time tRWMH(10000,x0,270,sigma,0.1)
@time RWMH(10000,x0,270,sigma,0.1)
@btime RWMH(10000,x0,270,sigma,0.1)
@benchmark tRWMH(1000,x0,270,sigma,0.1)
@benchmark t2RWMH(1000,x0,270,sigma,0.1)