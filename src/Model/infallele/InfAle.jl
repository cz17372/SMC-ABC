module InfAle

N = 50

using Distributions, Random

function disrand(n,lower,upper)
    return floor.(Int64,rand(Uniform(lower,upper),n))
end

function disrand_latents(val,lower,upper)
    range_x = upper - lower + 1
    return floor(Int64,val*range_x+lower)
end

function urn_example(ss,theta)
    mut_type = 1
    vec = zeros(Int64,ss)
    vec[1] = mut_type
    vec[2] = vec[1]
    nlin = 2
    while nlin < ss
        rr = rand()
        pmut = theta/(theta+nlin-1)
        ic = disrand(1,1,nlin)[1]
        if pmut > rr
            mut_type += 1
            vec[ic] = mut_type
            #println("mutation at lineage ",ic)
      
        
        
        else
            vec[nlin+1] = vec[ic]
            nlin+=1
            #println("coalescence on at lineage ",ic)
        end
    end
    return vec
end
function Sobs(lvec)
    return length(unique(lvec))
end

function urn_latents(ss,theta,latents)
    nlat = length(latents)
    lid  = 1
    if lid > nlat
        error("Not enough latents")
    end
    mut_type = 1
    vec = zeros(Int64,ss)
    vec[1] = mut_type
    vec[2] = vec[1]
    nlin = 2
    itrip = false
    while nlin < ss
        rr = itrip ? rand() : latents[lid]
        lid += 1
        itrip = lid > nlat
        pmut = theta/(theta+nlin-1)
        ic = disrand_latents(itrip ? rand() : latents[lid], 1, nlin)
        lid += 1
        itrip = lid > nlat
        if pmut > rr
            mut_type += 1
            vec[ic] = mut_type
        else
            vec[nlin+1] = vec[ic]
            nlin += 1
        end
    end
    return (nlat = lid, obs = vec)
end

U(u) = sum(logpdf(Uniform(0,1),u))

function GetPar()
    return rand(Uniform(0,10))
end

function GenSeed(N)
    return rand(N)
end


function Simulator(N)
    theta = GenPar()
    return urn_example(N,theta)
end


function ConSimulator(N,theta)
    return urn_example(N,theta)
end


function ϕ(u,theta;N)
    return urn_latents(N,theta,u)
end

function Ψ(u;N)
    theta = 10*u[1]
    return urn_latents(N,theta,u[2:end])
end


    



end

function SMC(N::Integer,sobs,ss,L;InitStep=0.2,MaxStep=1.0,MinStep=0.1,MinProb=0.2,IterScheme="Adaptive",InitIter=5,PropParMoved=0.99,TolScheme="unique",η=0.9,TerminalTol=1.0,TerminalProb=0.01,Parallel=true,gc = true)
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
        U[1][:,i] = rand(L)
        DISTANCE[i,1] = abs(Ψ(U[1][:,i],ss=ss)-sobs)
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
        println(length(index))
        ### ABC-MCMC exploration for alive particles 
        if Parallel
            v = Threads.@threads for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],sobs=sobs,δ=StepSize[end],L=A,ss=ss)
                DISTANCE[index[i],t+1] = abs(Ψ(U[t+1][:,index[i]],ss=ss)-sobs)
            end
        else
            v = for i = 1:length(index)
                U[t+1][:,index[i]],IndividualAcceptedNum[index[i]] = MCMC(K[t],U[t][:,ANCESTOR[index[i],t]],EPSILON[t+1],sobs=sobs,δ=StepSize[end],L=A,ss=ss)
                DISTANCE[index[i],t+1] = abs(Ψ(U[t+1][:,index[i]],ss=ss)-sobs)
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