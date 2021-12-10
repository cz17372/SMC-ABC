using Distributions, Plots, StatsPlots, Random, LinearAlgebra
theme(:ggplot2)

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

function ϕ(u,theta;ss)
    data = urn_latents(ss,theta,u)
    return Sobs(data.obs)
end
function Ψ(u;ss)
    theta = 10*u[1]
    data = urn_latents(ss,theta,u[2:end])
    return Sobs(data.obs)
end
p(u) = sum(logpdf(Uniform(0,1),u))

function MCMC(N,u0,ϵ;sobs,δ,L,ss)
    oldu = u0
    Ind = 0
    d = length(u0)
    Seed = rand(Normal(0,1),d,N)
    PropMove = δ * L * Seed
    for n = 1:N
        newu = oldu .+ PropMove[:,n]
        if log(rand(Uniform(0,1))) < p(newu) - p(oldu)
            if abs(Ψ(newu,ss=ss) - sobs) < ϵ
                oldu = newu
                Ind += 1
            end
        end
    end
    return (oldu,Ind)
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
            push!(EPSILON,floor(max(quantile(unique(DISTANCE[ANCESTOR[:,t],t]),η),TerminalTol)))
        elseif TolScheme == "ess"
            push!(EPSILON,floor(max(quantile(DISTANCE[ANCESTOR[:,t],t],η),TerminalTol)))
        end
        ### Calculate the weight for the next iteration
        WEIGHT = hcat(WEIGHT,(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1]))
        push!(ESS,1/sum(WEIGHT[:,end].^2))
        println("SMC Step: ", t)
        push!(UniqueStartingPoints,length(unique(DISTANCE[ANCESTOR[findall(WEIGHT[:,t+1] .> 0),t],t])))
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", UniqueStartingPoints[end])
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



Random.seed!(17372)
theta = 2.0
sobs = Sobs(urn_example(50,theta))


R100 = SMC(5000,sobs,50,101,η=0.8,TerminalTol=2.0,TerminalProb=0.0,InitIter=1000,TolScheme="ess")
Index = findall(R100.WEIGHT[:,end] .> 0)
theta100 = R100.U[end][1,Index]*10
R50 = SMC(5000,sobs,50,51,η=0.8,TerminalTol=1.0,TerminalProb=0.0,InitIter=1000,TolScheme="ess")
Index = findall(R50.WEIGHT[:,end] .> 0)
theta50 = R50.U[end][1,Index]*10
R150 = SMC(5000,sobs,50,151,η=0.8,TerminalTol=1.0,TerminalProb=0.0,InitIter=1000,TolScheme="ess")
Index = findall(R150.WEIGHT[:,end] .> 0)
theta150 = R150.U[end][1,Index]*10
R200 = SMC(5000,sobs,50,201,η=0.8,TerminalTol=1.0,TerminalProb=0.0,InitIter=1000,TolScheme="ess")
Index = findall(R200.WEIGHT[:,end] .> 0)
theta200 = R200.U[end][1,Index]*10



true_posterior = zeros(10000)
n = 1
trial = 0
while n <= 10000
    cand = rand(Uniform(0,10))
    trial += 1
    s = Sobs(urn_example(50,cand))
    if s == sobs
        true_posterior[n] = cand
        n += 1
    end
end

epsilon = 2
abc_posterior = zeros(10000)
n = 1
trial = 0
while n <= 10000
    cand = rand(Uniform(0,10))
    trial += 1
    s = Sobs(urn_example(50,cand))
    if abs(s-sobs) < epsilon
        abc_posterior[n] = cand
        n += 1
    end
end
density(true_posterior,label="true posterior")
density!(abc_posterior,label="abc posterior")
density!(theta100,label="L-ABC-SMC posterior (100 latents)")
density!(theta50,label="L-ABC-SMC posterior (50 latents)")
density!(theta150,label="L-ABC-SMC posterior (150 latents)")
vline!([2.0],label="",color=:grey,linestyle=:dash,linewidth=2)
savefig("result_mark.pdf")