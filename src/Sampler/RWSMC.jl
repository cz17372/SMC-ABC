module RWSMC
using Distributions, LinearAlgebra


function MCMC(N,u0,ϵ;y,δ,L,model::Module,Dist)
    oldu = u0
    Ind = 0
    d = length(u0)
    Seed = rand(Normal(0,1),d,N)
    PropMove = δ * L * Seed
    for n = 1:N
        newu = oldu .+ PropMove[:,n]
        if log(rand(Uniform(0,1))) < model.U(newu) - model.U(oldu)
            if Dist(model.Ψ(newu),y) < ϵ
                oldu = newu
                Ind += 1
            end
        end
    end
    return (oldu,Ind)
end


function SMC(N::Integer,y::Vector,L::Integer,model::Module,Dist;InitStep=0.2,MaxStep=1.0,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01,Parallel=true,gc = true)
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
        U[1][:,i] = model.GenSeed(L)
        DISTANCE[i,1] = Dist(model.Ψ(U[1][:,i]),y)
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
        if Parallel
            v = Threads.@threads for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],y=y,δ=StepSize[end],L=A,model=model,Dist=Dist)
                DISTANCE[index[i],t+1] = Dist(model.Ψ(U[t+1][:,index[i]]),y)
            end
        else
            v = for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],y=y,δ=StepSize[end],L=A,model=model)
                DISTANCE[index[i],t+1] = Dist(mod.Ψ(U[t+1][:,index[i]]),y)
            end
        end
        if gc
            GC.gc()
        end
        push!(UniqueParticles,length(unique(DISTANCE[findall(WEIGHT[:,t+1].>0),t+1])))
        ### Estimate the acceptance probability for the ABC_MCMC algorithm
        push!(AcceptanceProb,mean(IndividualAcceptedNum[index])/K[end])
        println("Average Acceptance Probability is ", AcceptanceProb[end])
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
        
        println("The step size used in the next SMC iteration is ",StepSize[end])
        print("\n\n")
        if EPSILON[end] == TerminalTol
            break
        end
    end
    return (U=U,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints) 
end


end