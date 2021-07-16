module DelMoralSMCABC
using LinearAlgebra: cholcopy
using Distributions, LinearAlgebra

f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

"""
    Simulate(N,θ,d)
Sample one or more sets of pseudo-observations from the underlying generative model.

# Arguments
- `N::Int`: the number of sets of pseudo-observations simulated
- `θ::Vector{Float}`: the value of parameter(s) used to simulate pseudo-observations
- `d::Int`: the number of pseudo observations simulated from the generator.
"""
function Simulate(N,θ,d)
    Output = zeros(N,d)
    for n = 1:N
        Output[n,:] = f.(rand(Normal(0,1),d),θ=10.0*θ)
    end
    return Output
end

"""
    SimulateOne(θ,d)
Generate one set of pseudo observations from the underlying generative model.

# Arguments
- `θ::Vector{Float}`: the value of parameter(s) used to simulate pseudo-observations
- `d::Int`: the number of pseudo observations simulated from the generator. 
"""
function SimulateOne(θ,d)
    return f.(rand(Normal(0,1),d),θ=10.0*θ)
end


"""
    ABC_MCMC(N,θ0,x0;y,ϵ,δ,Σ)
Perform the ABC_MCMC algorithm targeting the ABC density defined by the hard kernel and tolerance ϵ. 

# Arguments
- `N::Int`: the number of MCMC steps performed
- `θ0:Vector{Float}`: initial parameter(s) for the MCMC exploration
- `x0:Vector{Float}`: initial pseudo-observations associated with `θ0`
- `y:Vector{Float}`: the actual observations, should have the same size as `x0`
- `ϵ::Float`: the tolerance level for the ABC kernel
- `δ::Float`: the stepsize for the Metropolis-Hastings proposals
- `Σ::Float`: the covariance used to propose new parameter values
# Returns
The function will return the last set of parameters at the end of `N` MH steps, the last set of pseudo-observations and the number of accepted proposals at among `N` MH steps.
"""
function ABC_MCMC(N,θ0,x0;y,ϵ,δ,L)
    d = length(x0)
    oldθ = θ0
    oldx = x0;
    Ind = 0
    Seeds = rand(Normal(0,1),length(θ0),N)
    Proposals = δ * L * Seeds
    for n = 1:N
        newθ = oldθ .+ Proposals[:,n]
        if all(0.0 .< newθ .< 1.0)
            newx = SimulateOne(newθ,d)
            if norm(newx .- y) < ϵ
                oldx = newx
                oldθ = newθ
                Ind += 1
            end
        end
    end
    return (oldθ, oldx, Ind)
end


"""
    SMC(N,y;InitStep,MinStep,MinProb,IterScheme,InitIter,TolScheme,η,TerminalTol,TerminalProb)
ABC-SMC algorithm using ABC-MCMC to move the particles. Only one set of pseudo-observations is simulated at each MCMC step. 

# Arguments
- `N::Int`: the number of particles for the SMC sampler
- `y::Vector{Float}`: the actual observations
- `InitStep::Float`: initial step size for the ABC-MCMC algorithm
- `MinStep::Float`: the minimum step size that can be used in ABC-MCMC exploration within ABC-SMC algorithm
- `MinProb::Float`: the minimum acceptance probability ABC-MCMC algorithm should achieve. This is useful when the step size is adaptively tuned.
- `IterScheme::String`: either "Fixed" or "Adaptive". When "Fixed" is chosen, the number of ABC-MCMC iteration is fixed accross the algorithm. When "Adaptive" is chosen, the number of ABC-MCMC iteration will then be adaptively chosen so that `PropParMoved` of the particles will be moved after the ABC-MCMC exploration
- `InitIter::Int`: the initial number of iterations used in ABC-MCMC algorithm
- `PropParMoved::Float`: the proportion of particles that should be moved after the ABC-MCMC algorithm if "Adaptive" scheme is chosen for `IterScheme`
- `TolScheme::String`: the adaptive scheme used to adaptively choose the tolerance levels. Two there are two choices at the moment: "ess" and "unique". If "ess" is chosen, then the next tolerance level will be chosen such that the number of alive particles will be `η`*`N`. If "unique" is chosen, then the next tolerance level will be chosen such that the number of unique alive particles will be `η` * Nu, where Nu is the number of unique particles after resampling.
- `η::Float`: the proportion parameter used in choosing the tolerance levels.
- `TerminalTol::Float`: the terminal tolerance level. 
- `TerminalProb::Float`: the terminal acceptance probability

# Returns

"""
function SMC(N,y;InitStep=0.1,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01)
    ### Initialisation ###
    L = length(y)
    # U - particles for the parameters, each coloumn represents one particle
    U = Array{Matrix{Float64},1}(undef,0)
    push!(U,zeros(4,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int64,N,0)
    K = zeros(Int64,1); K[1] = InitIter;
    X = Array{Matrix{Float64},1}(undef,0)
    # X - pseudo-observations associated with each set of parameters stored in U
    # each coloumn represents o set of pseudo-observations
    push!(X,zeros(L,N))
    IndividualAcceptedNum = zeros(N); 
    AcceptanceProb = zeros(1); 
    # Set the first acceptance probability to be 1 to make the while loop work
    AcceptanceProb[1] = 1.0;
    StepSize = zeros(1); StepSize[1] = InitStep
    UniqueParticles = zeros(0)
    UniqueStartingPoints = zeros(0)
    ESS = zeros(0)
    ### Simulate the initial parameters & pseudo observations from prior ###
    for i = 1:N
        U[1][:,i] = rand(Uniform(0,1),4)
        X[1][:,i] = SimulateOne(U[1][:,i],L)
        DISTANCE[i,1] = norm(X[1][:,i] .- y)
    end
    push!(UniqueParticles,length(unique(DISTANCE[:,1])))
    WEIGHT[:,1] .= 1.0/N; t = 0;
    push!(ESS,1/sum(WEIGHT[:,end].^2))
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    timevec = zeros(0)
    ### ABC-SMC Part ### 
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
        push!(U,zeros(4,N)); push!(X,zeros(L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        ### ABC-MCMC exploration for alive particles 
        v = @timed Threads.@threads for i = 1:length(index)
            U[t+1][:,index[i]],X[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = ABC_MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],X[t][:,ANCESTOR[index[i],t]],y=y,ϵ=EPSILON[t+1],δ=StepSize[end],L=A)
            DISTANCE[index[i],t+1] = norm(X[t+1][:,index[i]] .- y)
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
    return (U=U,X=X,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],time=timevec,ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
end
end



