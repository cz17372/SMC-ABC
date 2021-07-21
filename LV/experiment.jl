using Distributions, Random, JLD2

println("Enter the Seeds for simulating the artificial dataset:")
seed = readline()
println("Enter the size of the dataset:")
L = readline()
seed = parse(Int64,seed)
L    = parse(Int64,L)

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
function SimulateOne(N)
    u = rand(4+N)
    return ϕ(u)
end

function ϕ(u)
    θ = -2.0 .+ 3.0*quantile(Normal(0,1),u[1:4])
    z = quantile(Normal(0,1),u[5:end])
    return f(z,θ=θ)
end

Random.seed!(seed);
θstar = log.([0.4,0.005,0.05,0.001]);
uθ = cdf(Normal(-2,3.0),θstar)
uz = rand(L)
ustar = [uθ;uz]
ystar = ϕ(ustar)

println("Choose the ABC-SMC method used:")
println("1. Standard ABC-SMC    2. Random-Walk ABC-SMC")
choice = readline()
choice = parse(Int64,choice)

if choice == 1
    include("DelMoralABCSMC.jl")
    println("The standard ABC-SMC method is used....")
    println("Enter the number of particles used:")
    N = readline()
    N = parse(Int64,N)
    println("Enter the initial step size:")
    InitStep = readline() 
    InitStep = parse(Float64,InitStep)
    println("Enter the minimum step size can be used:")
    MinStep = readline()
    MinStep = parse(Float64,MinStep)
    println("Enter the minimum probability for ABC-MCMC exploration:")
    MinProb = readline()
    MinProb = parse(Float64,MinProb)
    println("Choose the how the number of MCMC steps are determined:")
    println("1. Fixed No. of MCMC steps     2. Adaptive Number of MCMC steps")
    IterSchemeChoice = readline();
    if IterSchemeChoice == "1"
        IterScheme = "Fixed"
    elseif IterSchemeChoice == "2"
        IterScheme = "Adaptive"
    end
    println("Choose the initial no. of MCMC steps:")
    InitIter = readline(); InitIter = parse(Int64,InitIter);
    println("Choose the proportion of particles that should be moved by the MCMC algorithm:")
    PropParMoved = readline()
    PropParMoved = parse(Float64,PropParMoved)
    println("Choose the criterion used to choose tolerance:")
    println("1. ESS     2. Unique Number of Particles")
    TolSchemeChoice = readline();
    if TolSchemeChoice == "1"
        TolScheme = "ess"
    elseif TolSchemeChoice == "2"
        TolScheme = "unique"
    end
    println("Choose the proportion parameter used for choosing tolerance:")
    η = readline()
    η = parse(Float64,η)
    println("Choose the terminal tolerance:")
    TerminalTol = readline();
    TerminalTol = parse(Float64,TerminalTol)
    println("Choose the terminal probability:")
    TerminalProb = readline() 
    TerminalProb = parse(Float64,TerminalProb)

    println("Eneter the number of replica needed:")
    m = readline(); m = parse(Int64,m)

    EPSILON = Array{Any,1}(undef,m)
    K       = Array{Any,1}(undef,m)
    AcceptanceProb = Array{Any,1}(undef,m)
    U       = Array{Any,1}(undef,m)
    Time    = Array{Any,1}(undef,m)
    ESS     = Array{Any,1}(undef,m)
    StepSize = Array{Any,1}(undef,m)
    UniqueStartingPoints = Array{Any,1}(undef,m)
    UniqueParticles     = Array{Any,1}(undef,m)
    for i = 1:m
        R = DelMoralSMCABC.SMC(N,ystar,InitStep=InitStep,MinStep=MinStep,MinProb=MinProb,IterScheme=IterScheme,InitIter=InitIter,PropParMoved=PropParMoved,TolScheme=TolScheme,η=η,TerminalTol=TerminalTol,TerminalProb=TerminalProb)
        EPSILON[i] = R.EPSILON
        K[i]       = R.K 
        AcceptanceProb[i]       = R.AcceptanceProb
        Index      = findall(R.WEIGHT[:,end] .> 0)
        U[i]       = R.U[end][:,Index]
        Time[i] = R.time
        StepSize[i] = R.StepSize
        ESS[i] = R.ESS
        UniqueParticles[i] = R.UniqueParticles
        UniqueStartingPoints[i] = R.UniqueStartingPoints
    end
    Information = (seed = seed, L=L, Method="Standard ABC-SMC",N=N,InitStep=InitStep,MinStep=MinStep,MinProb=MinProb,IterScheme = IterScheme,InitIter=InitIter,PropParMoved=PropParMoved,TolScheme=TolScheme,η=η,TerminalTol=TerminalTol,TerminalProb=TerminalProb)
    Results = (Information=Information, EPSILON=EPSILON,K=K,AcceptanceProb=AcceptanceProb,U=U,Time=Time,ESS=ESS,StepSize=StepSize,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
    println("The name for the result file:")
    name = readline()
    @save name Results
elseif choice == 2
    include("RW.jl")
    println("The RandomWalk ABC-SMC method is used....")
    println("Enter the number of particles used:")
    N = readline();N = parse(Int64,N)
    println("Enter the initial step size:")
    InitStep = readline(); InitStep = parse(Float64,InitStep);
    println("Enter the minimum step size can be used:")
    MinStep = readline(); MinStep = parse(Float64,MinStep);
    println("Enter the minimum probability for ABC-MCMC exploration:")
    MinProb = readline();MinProb = parse(Float64,MinProb);
    println("Choose the how the number of MCMC steps are determined:")
    println("1. Fixed No. of MCMC steps     2. Adaptive Number of MCMC steps")
    IterSchemeChoice = readline();
    if IterSchemeChoice == "1"
        IterScheme = "Fixed"
    elseif IterSchemeChoice == "2"
        IterScheme = "Adaptive"
    end
    println("Choose the initial no. of MCMC steps:")
    InitIter = readline(); InitIter = parse(Int64,InitIter);
    println("Choose the proportion of particles that should be moved by the MCMC algorithm:")
    PropParMoved = readline(); PropParMoved = parse(Float64,PropParMoved)
    println("Choose the criterion used to choose tolerance:")
    println("1. ESS     2. Unique Number of Particles")
    TolSchemeChoice = readline();
    if TolSchemeChoice == "1"
        TolScheme = "ess"
    elseif TolSchemeChoice == "2"
        TolScheme = "unique"
    end
    println("Choose the proportion parameter used for choosing tolerance:")
    η = readline(); η = parse(Float64,η)
    println("Choose the terminal tolerance:")
    TerminalTol = readline(); TerminalTol = parse(Float64,TerminalTol);
    println("Choose the terminal probability:")
    TerminalProb = readline(); TerminalProb = parse(Float64,TerminalProb)

    println("Eneter the number of replica needed:")
    m = readline(); m = parse(Int64,m)

    EPSILON = Array{Any,1}(undef,m)
    K       = Array{Any,1}(undef,m)
    AcceptanceProb = Array{Any,1}(undef,m)
    U       = Array{Any,1}(undef,m)
    Time    = Array{Any,1}(undef,m)
    ESS     = Array{Any,1}(undef,m)
    StepSize = Array{Any,1}(undef,m)
    UniqueStartingPoints = Array{Any,1}(undef,m)
    UniqueParticles     = Array{Any,1}(undef,m)
    for i = 1:m
        R = RW.SMC(N,ystar,InitStep=InitStep,MinStep=MinStep,MinProb=MinProb,IterScheme=IterScheme,InitIter=InitIter,PropParMoved=PropParMoved,TolScheme=TolScheme,η=η,TerminalTol=TerminalTol,TerminalProb=TerminalProb)
        EPSILON[i] = R.EPSILON
        K[i]       = R.K 
        AcceptanceProb[i]       = R.AcceptanceProb
        Index      = findall(R.WEIGHT[:,end] .> 0)
        U[i]       = R.U[end][:,Index]
        Time[i] = R.time
        StepSize[i] = R.StepSize
        ESS[i] = R.ESS
        UniqueParticles[i] = R.UniqueParticles
        UniqueStartingPoints[i] = R.UniqueStartingPoints
    end
    Information = (seed = seed, L=L, Method="RW ABC-SMC",N=N,InitStep=InitStep,MinStep=MinStep,MinProb=MinProb,IterScheme = IterScheme,InitIter=InitIter,PropParMoved=PropParMoved,TolScheme=TolScheme,η=η,TerminalTol=TerminalTol,TerminalProb=TerminalProb)
    Results = (Information=Information,EPSILON=EPSILON,K=K,AcceptanceProb=AcceptanceProb,U=U,Time=Time,ESS=ESS,StepSize=StepSize,UniqueParticles=UniqueParticles,UniqueStartingPoints=UniqueStartingPoints)
    println("The name for the result file:")
    name = readline()
    @save name Results
end
