using Distributions, Plots
using ForwardDiff
using Random
using LinearAlgebra
using Flux
using LinearAlgebra
function ϕ(u;θ)
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
function SimulateOne(θ,N)
    u = rand(Normal(0,1),2*N)
    return ϕ(u,θ=θ)
end
function D(θ,u;y)
    return norm(ϕ(u,θ=θ) .- y)
end
function U(θ)
    return sum(logpdf.(Normal(-2.0,3.0),θ))
end
function ABC_MCMC(N,θ0,x0;y,ϵ,δ,Σ)
    d = length(x0) ÷ 2
    oldθ = θ0
    oldx = x0;
    Ind = 0
    for n = 2:(N+1)
        newθ = rand(MultivariateNormal(oldθ,δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newθ) - U(oldθ)
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

function ABC_MCMC2(N,θ0,x0;y,ϵ,δ,Σ)
    d = length(x0) ÷ 2
    Output = zeros(N+1,length(θ0))
    Output[1,:] = θ0
    oldx = x0;
    Ind = 0
    for n = 2:(N+1)
        newθ = rand(MultivariateNormal(Output[n-1,:],δ^2*Σ))
        if log(rand(Uniform(0,1))) < U(newθ) - U(Output[n-1,:])
            newx = SimulateOne(newθ,d)
            if norm(newx .- y) < ϵ
                oldx = newx
                Output[n,:] = newθ
                Ind += 1
            else
                Output[n,:] = Output[n-1,:]
            end
        else
            println("Reject")
            Output[n,:] = Output[n-1,:]
        end
    end
    return (Output, oldx, Ind)
end

function SMC(N,y;InitStep,MinStep,MinProb,IterScheme,InitIter,PropParMoved,TolScheme,η,TerminalTol,TerminalProb=0.01)
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
        U[1][:,i] = rand(Uniform(0,10),4)
        X[1][:,i] = SimulateOne(U[1][:,i],L ÷2)
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
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Metropolis-Hastings...")
        push!(U,zeros(4,N)); push!(X,zeros(L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        ### ABC-MCMC exploration for alive particles 
        v = @timed Threads.@threads for i = 1:length(index)
            U[t+1][:,index[i]],X[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = ABC_MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],X[t][:,ANCESTOR[index[i],t]],y=y,ϵ=EPSILON[t+1],δ=StepSize[end],Σ=Σ)
            GC.safepoint()
            DISTANCE[index[i],t+1] = norm(X[t+1][:,index[i]] .- y)
        end
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
        if (AcceptanceProb[end] < MinProb) & (StepSize[end] > MinStep)
            push!(StepSize,exp(log(StepSize[end]) + 0.5*(AcceptanceProb[end] - MinProb)))
        else
            push!(StepSize,StepSize[end])
        end
        println("Average Acceptance Probability is ", AcceptanceProb[t])
        println("The step size used in the next SMC iteration is ",StepSize[end])
        print("\n\n")
    end
    return (U=U,X=X,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,AcceptanceProb=AcceptanceProb,K=K[1:end-1],StepSize=StepSize[1:end-1],time=timevec,ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
end



Random.seed!(17372)
θstar = log.([0.4,0.005,0.05,0.001])
ustar = rand(Normal(0,1),100)
ystar = ϕ(ustar,θ=θstar)


R = SMC(1000,ystar,InitStep=0.3,MinStep=0.1,MinProb=0.1,IterScheme="Adaptive",InitIter=100,PropParMoved=0.99,TolScheme="unique",η=0.99,TerminalTol=10,TerminalProb=0.01)
Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]

using Plots, StatsPlots
density(X[1,:]); vline!([log(0.4)])
density(X[2,:]); vline!([log(0.005)])
density(X[3,:]);vline!([log(0.05)])
density(X[4,:]);vline!([log(0.001)])

plot(ystar,color=:red,linewidth=2.0)
for i = 1:length(Index)
    plot!(R.X[end][:,Index[i]],label="",linewidth=0.1,color=:grey)
end
current()

