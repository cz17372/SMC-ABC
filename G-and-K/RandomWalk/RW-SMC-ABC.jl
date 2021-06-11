module RandomWalk

using LinearAlgebra, Distributions
using Plots, StatsPlots
theme(:ggplot2)
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;


function Dist(ξ;y)
    return norm(f.(ξ[5:end],θ=ξ[1:4]) .- y)
end


#=
function Dist(ξ;y)
    return norm(sort(f.(ξ[5:end],θ=ξ[1:4])) .- sort(y))
end
=#

#=
function Dist(ξ;y)
    quantile_y = quantile(y,1/8*(1:8))
    x = f.(ξ[5:end],θ =ξ[1:4])
    quantile_x = quantile(x,1/8*(1:8))
    xA = quantile_x[4]
    yA = quantile_y[4]
    xB = quantile_x[6] - quantile_x[2]
    yB = quantile_y[6] - quantile_y[2]
    xg = (quantile_x[6] + quantile_x[2] - 2*quantile_x[4])/xB
    yg = (quantile_y[6] + quantile_y[2] - 2*quantile_y[4])/yB
    xk = (quantile_x[7] - quantile_x[5] + quantile_x[3] - quantile_x[1])/xB
    yk = (quantile_y[7] - quantile_y[5] + quantile_y[3] - quantile_y[1])/yB
    return norm([xA,xB,xg,xk] .- [yA,yB,yg,yk])
end
=#
C(ξ;ϵ,y) = Dist(ξ,y=y) - ϵ

function logpi(ξ;ϵ,y)
    if C(ξ,ϵ=ϵ,y=y) < 0
        logpdf_θ = sum(logpdf.(Uniform(0,10),ξ[1:4]))
        logpdf_z = sum(logpdf.(Normal(0,1),ξ[5:end]))
        return logpdf_θ + logpdf_z
    else
        return -Inf
    end
end

function RWMH(N,x0,ϵ;y,δ,Σ)
    X = zeros(N,length(x0))
    X[1,:] = x0
    AcceptedNum = 0
    for n = 2:N
        #xcand = X[n-1,:] .+ δ*L*rand(Normal(0,1),D)
        xcand = rand(MultivariateNormal(X[n-1,:],δ^2*Σ))
        α = min(0,logpi(xcand,ϵ=ϵ,y=y)-logpi(X[n-1,:],ϵ=ϵ,y=y))
        if log(rand(Uniform(0,1))) < α
            X[n,:] = xcand
            AcceptedNum += 1
        else
            X[n,:] = X[n-1,:]
        end
    end
    return (X[end,:],AcceptedNum)
end
function SMC(N,T,y;Threshold,δ,K0)
    NoData = length(y)
    U = zeros(4+NoData,N,T+1)
    EPSILON = zeros(T+1)
    DISTANCE = zeros(N,T+1)
    WEIGHT = zeros(N,T+1)
    ANCESTOR = zeros(Int,N,T)
    K = zeros(Int64,T+1)
    K[1] = K0
    for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        DISTANCE[i,1] = Dist(U[:,i,1],y=y)
    end
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    ParticleAcceptProb = zeros(N)
    MH_AcceptProb = zeros(T)
    for t = 1:T
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        println("SMC Step: ", t)
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("K = ", K[t])           
        Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        # L = cholesky(Σ).L
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        @time Threads.@threads for i = 1:length(index)
            U[:,index[i],t+1],ParticleAcceptProb[index[i]] = RWMH(K[t],U[:,ANCESTOR[index[i],t],t],EPSILON[t+1],y=y,δ=δ,Σ=Σ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = Dist(U[:,index[i],t+1],y=y)
        end
        MH_AcceptProb[t] = mean(ParticleAcceptProb[index])/K[t]
        K[t+1] = Int64(ceil(log(0.01)/log(1-MH_AcceptProb[t])))
        if MH_AcceptProb[t] < 0.25
            δ = exp(log(δ) + 0.3*(MH_AcceptProb[t] - 0.25))
        end
        println("Average Acceptance Probability is ", MH_AcceptProb[t])
        println("The step size used in the next SMC iteration is ",δ)
        print("\n\n")
    end
    return (U=U,DISTANCE=DISTANCE,WEIGHT=WEIGHT,EPSILON=EPSILON,ANCESTOR=ANCESTOR,AcceptanceProb = MH_AcceptProb, K = K)
end

function Epsilon(R;title="",label="",xlabel="Iteration",ylabel="log-Epsilon",color=:springgreen,new=true,figsize=(600,600))
    if new
        plot(log.(R.EPSILON),label=label,xlabel=xlabel,ylabel=ylabel,title=title,color=color,size=figsize)
    else
        plot!(log.(R.EPSILON),label=label,title=title,color=color)
    end
end

function ESS(R;title="",label="",xlabel="Iteration",ylabel="ESS",color=:springgreen,new=true,figsize=(600))
    N,T = size(R.WEIGHT)
    ESS = zeros(T)
    for i = 1:T
        ESS[i] = sum(R.WEIGHT[:,i] .> 0)
    end
    if new
        plot(ESS,xlabel=xlabel,ylabel=ylabel,title=title,label=label,color=color,size=figsize)
    else
        plot!(ESS,xlabel=xlabel,ylabel=ylabel,title=title,label=label,color=color)
    end
end

function UniqueParticle(R,title="",label="",xlabel="Iteration",ylabel="Unique Particle",color=:springgreen,new=true,figsize=(600,600))
    N,T = size(R.WEIGHT)
    UniqueParticles = zeros(T)
    for i = 1:T
        index = findall(R.DISTANCE[:,i] .> 0)
        UniqueParticles[i] = length(unique(R.DISTANCE[index,i]))
    end
    if new
        plot(UniqueParticles,xlabel=xlabel,ylabel=ylabel,title=title,label=label,color=color,size=figsize)
    else
        plot!(UniqueParticles,xlabel=xlabel,ylabel=ylabel,title=title,label=label,color=color)
    end
end


function epdf(R,Var,t;title="",label="",xlabel="",ylabel="Density",trueval=nothing,color=:springgreen,new=true,figsize=(600,600))
    index = findall(R.WEIGHT[:,t] .> 0)
    if new
        if trueval == nothing
            density(R.U[Var,index,t],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,size=figsize)
        else
            density(R.U[Var,index,t],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,size=figsize)
            vline!([trueval],color=:red,label="true value")
        end
    else
        if trueval == nothing
            density!(R.U[Var,index,t],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,size=figsize)
        else
            density!(R.U[Var,index,t],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color)
            vline!([trueval],color=:red,label="true value")
        end
    end
end

end