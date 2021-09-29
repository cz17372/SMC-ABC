module RESMC
using Distributions, Random, LinearAlgebra

function r(y)
    m = rem(y,2)
    if m < 0
        m += 2
    end
    if m < 1
        return m
    else
        return 2 - m
    end
end

function SliceSampling(x0;ϵ,ϕ,w)
    # sample velocity
    v = rand(Normal(0,1),length(x0))
    u = rand(Uniform(0,w)); a = -u; b= w-u
    NoSteps = 0
    while true
        z = rand(Uniform(a,b))
        x1 = r.(x0 .+ z*v)
        NoSteps += 1
        if ϕ(x1) < ϵ
            return (x1,NoSteps,abs(z))
        else
            if z < 0.0
                a = z
            else
                b = z
            end
        end
    end
end

function SMC(N,θstar;y,TerminalTol,g,Dist,η=0.5,Threshold=-Inf,w0=1.0,PrintRes=false,MT = true)
    ϕ(x) = Dist(g(x,θstar),y)
    L = length(y)
    X = Array{Matrix{Float64},1}(undef,0)
    push!(X,zeros(L,N))
    EPSILON = zeros(1); EPSILON[1] = Inf
    DISTANCE = zeros(N,1)
    PVec = zeros(0)
    ZVec = zeros(N)
    NumVec = zeros(N)
    WVec   = zeros(1); WVec[1] = w0
    AveNum = zeros(0)
    for i = 1:N
        X[1][:,i] = rand(L)
        DISTANCE[i,1] = ϕ(X[1][:,i])
    end
    t = 1
    while EPSILON[end] >= TerminalTol
        t += 1
        push!(EPSILON,max(quantile(unique(DISTANCE[:,t-1]),η),TerminalTol))
        Index = findall(DISTANCE[:,t-1] .<= EPSILON[t])
        push!(PVec,log.(length(Index)/N))
        if sum(PVec) < Threshold
            break
        end
        RIndex = sample(Index,N,replace=true)
        push!(X,zeros(L,N))
        DISTANCE = hcat(DISTANCE,zeros(N))
        if PrintRes
            println("SMC Step: ",t-1)
            println("EPSILON = ",EPSILON[t]," w = ",WVec[end])
        end
        if MT
            Threads.@threads for i = 1:N
                X[t][:,i],NumVec[i],ZVec[i] = SliceSampling(X[t-1][:,RIndex[i]],ϵ=EPSILON[t],ϕ=ϕ,w=WVec[end])
                DISTANCE[i,t] = ϕ(X[t][:,i])
            end
        else
            for i = 1:N
                X[t][:,i],NumVec[i],ZVec[i] = SliceSampling(X[t-1][:,RIndex[i]],ϵ=EPSILON[t-1],ϕ=ϕ,w=WVec[end])
                DISTANCE[i,t] = ϕ(X[t][:,i])
            end
        end
        push!(AveNum,mean(NumVec))
        #push!(WVec,w0)
        push!(WVec,min(1.0,2*findmax(ZVec)[1]))
        if PrintRes
            println("Average Number of Steps = ",AveNum[end])
        end
        if EPSILON[end] == TerminalTol
            break
        end
    end
    return (PVec=PVec,X=X,AveNum=AveNum,WVec=WVec,DISTANCE=DISTANCE,EPSILON=EPSILON)
end

function PMMH(θ0,M,N;y,model,Dist,ϵ,Σ,η=0.5,δ=2.562/4,MT=true,PR=false)
    theta = zeros(M+1,length(θ0))
    theta[1,:] = θ0
    llkvec = zeros(M+1)
    NumVec = zeros(M+1)
    R = SMC(N,theta[1,:],y=y,g=model.g,Dist=Dist,η=η,TerminalTol=ϵ,Threshold=-Inf,PrintRes=PR)
    llkvec[1] = sum(R.PVec)
    NumVec[1] = sum(R.AveNum * N)
    Accept = 0
    for n = 2:(M+1)
        newθ = rand(MultivariateNormal(theta[n-1,:],δ^2*Σ))
        if all(0.0 .< newθ .< 10.0)
            u = rand(Uniform(0,1))
            thres = log(u)+llkvec[n-1]
            R = SMC(N,newθ,y=y,η=η,g=model.g,Dist=Dist,TerminalTol=ϵ,Threshold=thres,MT=MT,PrintRes=PR)
            if sum(R.PVec) < thres
                theta[n,:] = theta[n-1,:]
                llkvec[n]  = llkvec[n-1]
                NumVec[n] = sum(R.AveNum * N)
                println(n, "No")
            else
                theta[n,:] = newθ
                llkvec[n]  = sum(R.PVec)
                NumVec[n]  = sum(R.AveNum * N)
                Accept += 1
                println(n,"Yes")
            end
        else
            theta[n,:] = theta[n-1,:]
            llkvec[n]  = llkvec[n-1]
            println(n,"No")
        end
    end
    return (theta=theta,NumVec = NumVec, llkvec=llkvec, Accprob = Accept/M)
end
end


