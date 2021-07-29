f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

function ϕ(u)
    θ = 10.0*u[1:4]
    z = quantile(Normal(0,1),u[5:end])
    return f.(z,θ=θ)
end
SampleOne(L) = rand(Uniform(0,1),4+L)
d(u) = norm(ϕ(u) .- ystar)
Euclidean(u;y) = norm(ϕ(u) .- y)
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
function SliceSampling(x0;ϵ,w,ϕ)
    v = rand(Normal(0,1),length(x0))
    u = rand(Uniform(0,w))
    a = -u; b= w - u
    num = 0
    while true
        z = rand(Uniform(a,b))
        x1 = r.(x0 .+ z*v)
        num += 1
        if ϕ(x1) < ϵ
            return (x1,num,z)
        else
            if z < 0
                a = z
            else
                b = z
            end
        end
    end
end

function SMC(N,y;InitStep=1.0,MinStep=0.0,TolScheme = "unique",η=0.9,TerminalTol = 1.0)
    d(u) = Euclidean(u,y=y)
    L = length(y)
    U = Array{Matrix{Float64},1}(undef,0)
    push!(U,zeros(4+L,N))
    EPSILON = zeros(1)
    DISTANCE = zeros(N,1)
    WEIGHT = zeros(N,1)
    ANCESTOR = zeros(Int64,N,0)
    NumVec = zeros(N)
    IterVec = zeros(0)
    ZVec   = zeros(N)
    StepSize = zeros(1); StepSize[1] = InitStep
    timevec = zeros(0)
    UniqueParticles = zeros(0)
    UniqueStartingPoints = zeros(0)
    ESS = zeros(0)
    for i = 1:N
        U[1][:,i] = SampleOne(L)
        DISTANCE[i,1] = d(U[1][:,i])
    end
    push!(UniqueParticles,length(unique(DISTANCE[:,1])))
    WEIGHT[:,1] .= 1.0/N
    push!(ESS,1/sum(WEIGHT[:,end].^2))
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    t = 0
    while EPSILON[end] > TerminalTol
        t += 1
        ANCESTOR = hcat(ANCESTOR,vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...));
        ### Choose Next Tolerance According to the Input Scheme ###
        if TolScheme == "unique"
            push!(EPSILON,quantile(unique(DISTANCE[ANCESTOR[:,t],t]),η))
        elseif TolScheme == "ess"
            push!(EPSILON,quantile(DISTANCE[ANCESTOR[:,t],t],η))
        end
        WEIGHT = hcat(WEIGHT,(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1]))
        push!(ESS,1/sum(WEIGHT[:,end].^2))
        println("SMC Step: ", t)
        push!(UniqueStartingPoints,length(unique(DISTANCE[ANCESTOR[:,t],t])))
        println("epsilon = ", round(EPSILON[t+1],sigdigits=5), " No. Unique Starting Point: ", length(unique(DISTANCE[ANCESTOR[:,t],t])))
        index = findall(WEIGHT[:,t+1] .> 0.0)
        println("Performing local Slice Sampling...")
        push!(U,zeros(4+L,N)); 
        DISTANCE = hcat(DISTANCE,zeros(N));
        v = @timed Threads.@threads for i = 1:length(index)
            U[t+1][:,index[i]],NumVec[index[i]],ZVec[index[i]] = SliceSampling(U[t][:,ANCESTOR[index[i],t]],ϵ=EPSILON[t+1],w=StepSize[end],ϕ=d)
            DISTANCE[index[i],t+1] = d(U[t+1][:,index[i]])
        end
        GC.gc()
        push!(UniqueParticles,length(unique(DISTANCE[findall(WEIGHT[:,t+1].>0),t+1])))
        push!(IterVec,mean(NumVec[index]))
        push!(timevec,v.time-v.gctime)
        println("Mean No. Steps for Slice Sampling is",mean(NumVec[index]))
        push!(StepSize,max(MinStep,min(1.0,10*findmax(abs.(ZVec[index]))[1])))
        println("The value of w in the next SMC step is",StepSize[end])
        print("\n\n")
    end
    return (U=U,EPSILON=EPSILON,DISTANCE=DISTANCE,WEIGHT=WEIGHT,ANCESTOR=ANCESTOR,WVEC=StepSize,time=timevec,ESS=ESS,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints,IterVec=IterVec)
end

R = SMC(10000,ystar,η=0.8,TerminalTol=3.0)
Index = findall(R.WEIGHT[:,end] .> 0)
X = R.U[end][:,Index]
density(X[1,:])

plot(R.IterVec)