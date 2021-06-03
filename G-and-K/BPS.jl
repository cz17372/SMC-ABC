using LinearAlgebra, Distributions, StatsPlots, Plots, Roots, ProgressMeter
using ForwardDiff:gradient
using Random
C(x) = norm(x) - 0.5
Σ = [1 2; 2 5]
μ = [0,0]
U(x) = logpdf(MultivariateNormal(μ,Σ),x)


object(k;x0,u0,C) = prod(x0 .+ k*u0)*prod(x0 .+ k*u0 .- 0.5)*C(x0 .+  k*u0)
prior_boundary(x0) = any([(abs.(x0) .< 1e-15);(abs.(x0 .- 0.5) .< 1e-15)])
get_prior_normal(x0) = normalize((abs.(x0) .< 1e-15) .+ (abs.(x0 .- 0.5) .< 1e-15))

function φ1(x0,u0,δ)
    output = x0
    working_delta = δ
    intermediate_x = copy(x0)
    intermediate_u = copy(u0)
    roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    while length(roots) > 0
        k = roots[1]
        intermediate_x = intermediate_x .+ k*intermediate_u
        output = hcat(output,intermediate_x)
        working_delta -= k
        if prior_boundary(intermediate_x)
            n = get_prior_normal(intermediate_x)
        else
            n = normalize(gradient(C,intermediate_x))
        end
        intermediate_u = intermediate_u .- 2*dot(intermediate_u,n)*n
        roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    end
    output = hcat(output,intermediate_x .+ working_delta * intermediate_u)
    return output[:,end], -intermediate_u
end


function b(x0::Vector{Float64},u0::Vector{Float64};gradFunc)
    n = normalize(gradient(gradFunc,x0))
    return u0 .- 2.0 * dot(u0,n) * n
end


function σ(x0,u0)
    return (x0,-u0)
end


function α1(x0,u0,δ)
    x1 = φ1(x0,u0,δ)[1]
    return min(0,U(x1)-U(x0))
end

function α2(x0,u0,x2,u2,δ)
    firstproprejectratio = log(1 - exp(α1(x2,u2,δ))) - log(1 - exp(α1(x0,u0,δ)))
    llkratio = U(x2) - U(x0)
    return min(0,firstproprejectratio + llkratio)
end

function DirectionRefresh(u0,δ,κ)
    p = exp(-κ*δ)
    ind = rand(Bernoulli(p))
    if ind == 1
        return u0
    else
        return normalize(rand(Normal(0,1),length(u0)))
    end
end


function BPS(N,x0,δ,κ)
    X = zeros(N,2)
    X[1,:] = x0
    u0 = normalize(rand(Normal(0,1),2))
    acc = 0
    @showprogress 1 "Computing.." for n = 2:N
        #println("Iteration ",n)
        #println("x0 = ",X[n-1,:])
        #println("u0 = ",u0)
        x1,u1 = φ1(X[n-1,:],u0,δ)
        #println("x1 = ",x1)
        #println("u1 = ",u1)
        if log(rand(Uniform(0,1))) < α1(X[n-1,:],u0,δ)
            acc += 1
            xhat = x1
            uhat = u1
        else
            xbound = x1; ubound = b(xbound,-u1,gradFunc=U)
            x2,u2  = φ1(xbound,ubound,δ)
            #println("x2 = ",x2)
            #println("u2 = ",u2)
            if log(rand(Uniform(0,1))) < α2(X[n-1,:],u0,x2,u2,δ)
                acc += 1
                xhat = x2
                uhat = u2
            else
                xhat = X[n-1,:]
                uhat = u0
            end
        end
        xhat,uhat = σ(xhat,uhat)
        u0 = DirectionRefresh(uhat,δ,κ)
        X[n,:] = xhat
    end
    return X,acc/(N-1)
end

x0 = [0.1,0.2]
u0 = normalize(rand(Normal(0,1),2))
X,u = test(x0,u0,0.8)

#X,acc = BPS(30000,x0,0.3,1.0)
t = collect(0:0.001:pi/2)
plot(0.5*cos.(t),0.5*sin.(t),size=(600,600),label="",color = :black,linewidth=2.0)
plot!([0,0],[0,0.5],label="",color = :black,linewidth=2.0)
plot!([0,0.5],[0,0],label="",color = :black,linewidth=2.0)
plot!(X[1,:],X[2,:],label="",color=:darkolivegreen)
scatter!(X[1,:],X[2,:],markersize=5.0,markerstrokewidth=0.0,label="",color=:darkolivegreen)
plot!([X[1,2],x0[1]+0.8*u0[1]],[X[2,2],x0[2]+0.8*u0[2]],label="",color=:red)
scatter!([x0[1]+0.8*u0[1]],[x0[2]+0.8*u0[2]],markersize=5.0,markerstrokewidth=0.0,label="",color=:red)
scatter!(trueX[:,1],trueX[:,2],markersize=0.5,markerstrokewidth=0.0,color=:red)


trueX = zeros(20000,2)
n = 1
while n <=20000
    cand = rand(MultivariateNormal(μ,Σ))
    if all(cand .> 0) && C(cand) <0
        trueX[n,:] = cand
        n += 1
    end
end
function test(x0,u0,δ)
    output = x0
    working_delta = δ
    intermediate_x = copy(x0)
    intermediate_u = copy(u0)
    roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    while length(roots) > 0
        k = roots[1]
        intermediate_x = intermediate_x .+ k*intermediate_u
        output = hcat(output,intermediate_x)
        working_delta -= k
        if prior_boundary(intermediate_x)
            n = get_prior_normal(intermediate_x)
        else
            n = normalize(gradient(C,intermediate_x))
        end
        intermediate_u = intermediate_u .- 2*dot(intermediate_u,n)*n
        roots = find_zeros(k->object(k,x0=intermediate_x,u0=intermediate_u,C=C),0,working_delta); roots = roots[roots.>0]
    end
    output = hcat(output,intermediate_x .+ working_delta * intermediate_u)
    return output, -intermediate_u
end