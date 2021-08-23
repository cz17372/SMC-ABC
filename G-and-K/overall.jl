using Distributions, LinearAlgebra, Random
f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;
function ϕ(u)
    θ = 10.0*cdf(Normal(0,1),u[1:4])
    return f.(u[5:end],θ=θ)
end

function MCMC(N,u0,ϵ;y,δ,L,ϕ)
    oldu = u0
    Ind = 0
    d = length(u0)
    Seed = rand(Normal(0,1),d,N)
    PropMove = δ * L * Seed
    for n = 1:N
        newu = oldu .+ PropMove[:,n]
        if log(rand(Uniform(0,1))) < U(newu) - U(oldu)
            if norm(ϕ(newu) .- y) < ϵ
                oldu = newu
                Ind += 1
            end
        end
    end
    return (oldu,Ind)
end

Dist(x,y) = norm(x .- y)


U(x) = sum(logpdf(Normal(0,1),x))

function SMC(N,y,L,ϕ,Dist;InitStep=0.1,MaxStep=1.0,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01,MultiThread=true,gc = true)
    ### Initialisation ###
    U = Array{Matrix{Float64},1}(undef,0)
    push!(U,zeros(L,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int64,N,0)
    K = zeros(Int64,1); K[1] = InitIter;
    IndividualAcceptedNum = zeros(N)
    AcceptanceProb = zeros(1)
    AcceptanceProb[1] = 1.0
    StepSize = zeros(1); StepSize[1] = InitStep;
    UniqueParticles = zeros(0)
    UniqueStartingPoints = zeros(0)
    ESS = zeros(0)
    ### Simulate Initial particles ###
    for i = 1:N
        U[1][:,i] = rand(Normal(0,1),L)
        DISTANCE[i,1] = Dist(ϕ(U[1][:,i]),y)
    end
    push!(UniqueParticles,length(unique(DISTANCE[:,1])))
    WEIGHT[:,1] .= 1.0/N
    push!(ESS,1/sum(WEIGHT[:,end].^2))
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    t = 0
    while AcceptanceProb[end] > TerminalProb
        t += 1
        ### Resampling Step ###
        ANCESTOR = hcat(ANCESTOR,zeros(Int64,N))
        ANCESTOR[:,t] = sample(findall(WEIGHT[:,t].>0),N,replace=true)
        ### Choose Next Tolerance According to the Input Scheme ###
        if TolScheme == "unique"
            push!(EPSILON,max(quantile(unique(DISTANCE[ANCESTOR[:,t],t]),η),TerminalTol))
        elseif TolScheme == "ess"
            push!(EPSILON,max(quantile(DISTANCE[ANCESTOR[:,t],t],η),TerminalTol))
        end
        ### Calculate the weight for the next iteration
        WEIGHT = hcat(WEIGHT,(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1]))
        push!(ESS,1/sum(WEIGHT[:,end].^2))
        println("SMC Step: ", t)
        push!(UniqueStartingPoints,length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("K = ", K[t])
        Σ = cov(U[t][:,findall(WEIGHT[:,t].>0)],dims=2) + 1e-8*I
        A = cholesky(Σ).L
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        push!(U,zeros(L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        ### ABC-MCMC exploration for alive particles 
        if MultiThread
            v = Threads.@threads for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],y=y,δ=StepSize[end],L=A,ϕ=ϕ)
                DISTANCE[index[i],t+1] = Dist(ϕ(U[t+1][:,index[i]]),y)
            end
        else
            v = for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],y=y,δ=StepSize[end],L=A,ϕ=ϕ)
                DISTANCE[index[i],t+1] = Dist(ϕ(U[t+1][:,index[i]]),y)
            end
        end
        if gc
            GC.gc()
        end
        push!(UniqueParticles,length(unique(DISTANCE[findall(WEIGHT[:,t+1].>0),t+1])))
        ### Estimate the acceptance probability for the ABC_MCMC algorithm
        push!(AcceptanceProb,mean(IndividualAcceptedNum[index])/K[end])
        if IterScheme=="Adaptive"
            push!(K,Int64(ceil(log(1-PropParMoved)/log(1-AcceptanceProb[end]))))
        elseif IterScheme == "Fixed"
            push!(K,InitIter)
        end
        ### Tune the step size ### 
        push!(StepSize,min(MaxStep,max(MinStep,exp(log(StepSize[end]) + 0.5*(AcceptanceProb[end] - MinProb)))))
        """
        if StepSize[end] > MinStep
            push!(StepSize,exp(log(StepSize[end]) + 0.5*(AcceptanceProb[end] - MinProb)))
        else
            push!(StepSize,StepSize[end])
        end
        """
        println("Average Acceptance Probability is ", AcceptanceProb[t])
        println("The step size used in the next SMC iteration is ",StepSize[end])
        print("\n\n")
        if EPSILON[end] == TerminalTol
            break
        end
    end
    return (U=U,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints) 
end


R = SMC(2000,ystar,length(ystar)+4,ϕ,Dist,TerminalTol=2.0)

function f(u;θ)
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

function ϕ(u)
    θ = -2.0 .+ 3.0*u[1:4]
    return f(u[5:end],θ=θ)
end
Random.seed!(17372);
θstar = log.([0.4,0.005,0.05,0.001]);
uθ = (θstar .+ 2.0)/3.0
uz = rand(Normal(0,1),100)
ustar = [uθ;uz]
ystar = ϕ(ustar)

R = SMC(2000,ystar,length(ystar)+4,ϕ,Dist,TerminalTol=5.0,η =0.8,gc=true)

Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
obs = mapslices(ϕ,X,dims=1)
plot(obs[1:50,1],color=:grey,linewidth=0.1,label="")
for n = 2:size(X)[2]
    plot!(obs[1:50,n],color=:grey,linewidth=0.1,label="")
end
for n = 1:size(X)[2]
    plot!(obs[51:100,n],color=:grey,linewidth=0.1,label="")
end
current()

plot!(ystar[1:50],color=:red,linewidht=3,label="")
plot!(ystar[51:end],color=:green,linewidht=3,label="")