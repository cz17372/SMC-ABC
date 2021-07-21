module RW
using Distributions, LinearAlgebra
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

Euclidean(u;y) = norm(ϕ(u) .- y)
SampleOne(L) = rand(Normal(0,1),4+L)
U(u) = sum(logpdf(Normal(0,1),u))
function MCMC(N,x0,ϵ;y,δ,L)
    oldx = x0
    Ind = 0
    d = length(x0)
    Seed = rand(Normal(0,1),d,N)
    PropMove = δ*L*Seed
    for n = 1:N
        newx = oldx .+ PropMove[:,n]
        # newx = rand(MultivariateNormal(oldx,δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newx) - U(oldx)
            if Euclidean(newx,y=y) < ϵ
                oldx = newx
                Ind += 1
            end
        end
    end
    return (oldx,Ind)
end
function SMC(N,y;InitStep=0.1,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01)
    ### Initialisation ###
    L = length(y)
    U = Array{Matrix{Float64},1}(undef,0)
    push!(U,zeros(4+L,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int64,N,0)
    K = zeros(Int64,1); K[1] = InitIter;
    IndividualAcceptedNum = zeros(N)
    AcceptanceProb = zeros(1)
    AcceptanceProb[1] = 1.0
    StepSize = zeros(1); StepSize[1] = InitStep;
    timevec = zeros(0)
    UniqueParticles = zeros(0)
    UniqueStartingPoints = zeros(0)
    ESS = zeros(0)
    ### Simulate Initial particles ###
    for i = 1:N
        U[1][:,i] = SampleOne(L)
        DISTANCE[i,1] = Euclidean(U[1][:,i],y=y)
    end
    push!(UniqueParticles,length(unique(DISTANCE[:,1])))
    WEIGHT[:,1] .= 1.0/N
    push!(ESS,1/sum(WEIGHT[:,end].^2))
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    t = 0
    while (AcceptanceProb[end] > TerminalProb) & (EPSILON[end] > TerminalTol)
        t += 1
        ### Resampling Step ###
        ANCESTOR = hcat(ANCESTOR,vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...));
        ### Choose Next Tolerance According to the Input Scheme ###
        if TolScheme == "unique"
            push!(EPSILON,quantile(unique(DISTANCE[ANCESTOR[:,t],t]),η))
        elseif TolScheme == "ess"
            push!(EPSILON,quantile(DISTANCE[ANCESTOR[:,t],t],η))
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
        push!(U,zeros(4+L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        ### ABC-MCMC exploration for alive particles 
        v = @timed Threads.@threads for i = 1:length(index)
            U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],y=y,δ=StepSize[end],L=A)
            DISTANCE[index[i],t+1] = Euclidean(U[t+1][:,index[i]],y=y)
        end
        GC.gc()
        push!(UniqueParticles,length(unique(DISTANCE[findall(WEIGHT[:,t+1].>0),t+1])))
        push!(timevec,v.time-v.gctime)
        ### Estimate the acceptance probability for the ABC_MCMC algorithm
        push!(AcceptanceProb,mean(IndividualAcceptedNum[index])/K[end])
        if IterScheme=="Adaptive"
            push!(K,Int64(ceil(log(1-PropParMoved)/log(1-AcceptanceProb[end]))))
        elseif IterScheme == "Fixed"
            push!(K,InitIter)
        end
        ### Tune the step size ### 
        push!(StepSize,max(MinStep,exp(log(StepSize[end]) + 0.5*(AcceptanceProb[end] - MinProb))))
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
    end
    (U=U,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],time=timevec,ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
end
end