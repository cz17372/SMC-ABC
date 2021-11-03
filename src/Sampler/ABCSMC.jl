module ABCSMC
using Distributions, Random, LinearAlgebra
function ABC_MCMC(N,θ0,x0;model,Dist,y,ϵ,δ,L)
    d = length(x0)
    oldθ = θ0;
    oldx = x0;
    Ind = 0
    Seeds = rand(Normal(0,1),length(θ0),N)
    Proposals = δ * L * Seeds

    for n = 1:N
        newθ = oldθ .+ Proposals[:,n]
        alpha = min(0,model.ptheta(newθ) - model.ptheta(oldθ))
        if log(rand()) < alpha
            newx = model.ConSimulator(d,newθ)
            if Dist(newx,y) < ϵ
                oldx = newx
                oldθ = newθ
                Ind += 1
            end
        end
    end

    return (oldθ,oldx,Ind)
end

function SMC(N,y,model,Dist;InitStep=0.1,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01,Parallel=true,gc=false)
    L = length(y)
    pardim = model.NoParam
    # U - particles for the parameters, each coloumn represents one particle
    U = Array{Matrix{Float64},1}(undef,0)
    push!(U,zeros(pardim,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int64,N,0)
    K = zeros(Int64,1); K[1] = InitIter;
    X = Array{Matrix{Float64},1}(undef,0)
    push!(X,zeros(L,N))
    IndividualAcceptedNum = zeros(N); 
    AcceptanceProb = zeros(1); 
    # Set the first acceptance probability to be 1 to make the while loop work
    AcceptanceProb[1] = 1.0;
    StepSize = zeros(1); StepSize[1] = InitStep
    UniqueParticles = zeros(0)
    UniqueStartingPoints = zeros(0)
    ESS = zeros(0)
    for i = 1:N
        U[1][:,i] = model.GenPar()
        X[1][:,i] = model.ConSimulator(L,U[1][:,i])
        DISTANCE[i,1] = Dist(X[1][:,i],y)
    end
    push!(UniqueParticles,length(unique(DISTANCE[:,1])))
    WEIGHT[:,1] .= 1.0/N; t = 0;
    push!(ESS,1/sum(WEIGHT[:,end].^2))
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
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
        push!(U,zeros(4,N)); push!(X,zeros(L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        ### ABC-MCMC exploration for alive particles 
        if Parallel
            Threads.@threads for i = 1:length(index)
                U[t+1][:,index[i]],X[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = ABC_MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],X[t][:,ANCESTOR[index[i],t]],model=model,Dist=Dist,y=y,ϵ=EPSILON[t+1],δ=StepSize[end],L=A)
                DISTANCE[index[i],t+1] = Dist(X[t+1][:,index[i]],y)
            end
        else
            for i = 1:length(index)
                U[t+1][:,index[i]],X[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = ABC_MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],X[t][:,ANCESTOR[index[i],t]],model=model,Dist=Dist,y=y,ϵ=EPSILON[t+1],δ=StepSize[end],L=A)
                DISTANCE[index[i],t+1] = Dist(X[t+1][:,index[i]],y)
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
        push!(StepSize,max(MinStep,exp(log(StepSize[end]) + 1.0*(AcceptanceProb[end] - MinProb))))
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
    return (U=U,X=X,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
end

end
