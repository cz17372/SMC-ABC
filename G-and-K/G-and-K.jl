using Distributions, Roots, ForwardDiff, LinearAlgebra, ProgressMeter,Plots, StatsPlots, Random


function Transform_Normal(z;par)
    # Define the G-and-K model using standar Normal distributions
    return par[1] + par[2]*(1+0.8*(1-exp(-par[3]*z))/(1+exp(-par[3]*z)))*((1+z^2)^par[4])*z
end


function inverse(x;par)
    f(z) = Transform_Normal(z,par=par)-x
    upp = 0.0; low = 0.0;
    while f(upp)*f(low)>0
        upp += 1.0
        low -= 1.0
    end
    return fzero(f,(low,upp))
end


function grad(z;par)
    ForwardDiff.derivative(x->Transform_Normal(x,par=par),z)
end



function Generate_Data(N;par,NoisyData = false,noise=nothing)
    z = rand(Normal(0,1),N)
    if NoisyData
        return Transform_Normal.(z,par=par) + rand(Normal(0,noise),N)
    else
        return Transform_Normal.(z,par=par)
    end
end



function SMC(N,T,data;Criterion="ESS",Threshold=0.8,NoisyData=false,noise=nothing,Method="New",scale=0.05)
    """
    N            : The number of particles at each SMC step
    T            : Number of SMC steps
    data         : The observations of the (noisy) g-and-k distributions
    Criterion    : The criterion used to choose the next temperature (i.e. ϵ_{t+1}) 
    """
    """
    Method == "New" implements the new SMC-ABC method. Instead of making purtabations on the 
    static paramters only and generating a new set of hidden states, the new method make 
    purtabations on both the static parameters and hidden states at each iteration
    """
    ϵ = zeros(T+1);
    # Define a matrix storing the weights
    W = zeros(N,T+1);
    # Define a matrix storing the ancestors
    A = zeros(Int,N,T);
    Distance  = zeros(N,T+1);

    if Method == "New"
        Particles = zeros(N,4+length(data),T+1);
    elseif Method == "Standard"
        Particles = zeros(N,4,T+1);
    end
    function dist(y,x)
        return sqrt(sum((sort(y) .- sort(x)).^2))
    end
    function f(xi;NoisyData,noise,Method,data)
        if Method == "New"
            par = xi[1:4]
            z   = xi[5:end]
            if NoisyData
                return Transform_Normal.(z,par=par) + rand(Normal(0,noise),length(z))
            else
                return Transform_Normal.(z,par=par)
            end
        elseif Method == "Standard"
            par = xi
            z   = rand(Normal(0,1),length(data))
            if NoisyData
                return Transform_Normal.(z,par=par) + rand(Normal(0,noise),length(z))
            else
                return Transform_Normal.(z,par=par)
            end
        end
    end
    function logPrior(xi;Method)
        if Method == "New"
            logparam = sum(logpdf.(Uniform(0,10),xi[1:4]))
            logz     = sum(logpdf.(Normal(0,1),xi[5:end]))
            return logparam + logz
        elseif Method == "Standard"
            logparam = sum(logpdf.(Uniform(0,10),xi[1:4]))
            return logparam
        end
    end
    function LocalMH(xi0,covariance,epsilon,y;Method,data)
        # sample the new candidate xi according to a random walk kernel
        newxi = rand(MultivariateNormal(xi0,covariance))
        # sample a Uniform(0,1) RV
        u = rand(Uniform(0,1))
        if log(u) >= logPrior(newxi,Method=Method)-logPrior(xi0,Method=Method)
            return xi0
        else
            x = f(newxi,NoisyData=NoisyData,noise=noise,Method = Method,data=data)
            if dist(y,x) < epsilon
                return newxi
            else
                return xi0
            end
        end
    end
        
    t = 0
    for n = 1:N
        if Method == "New"
            Particles[n,:,t+1] = [rand(Uniform(0,10),4);rand(Normal(0,1),length(data))]
        elseif Method == "Standard"
            Particles[n,:,t+1] = rand(Uniform(0,10),4)
        end
        Distance[n,t+1]    = dist(data,f(Particles[n,:,t+1],NoisyData=NoisyData,noise=noise,Method=Method,data=data))
    end
    ϵ[t+1] = findmax(Distance[:,t+1])[1]
    # Since we are sampling all the particles from prior, all the particles will have equal weights
    W[:,t+1] .= 1/N;
    @showprogress 1 "Computing..." for t = 1:T
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...)
        if Criterion == "ESS"
            ϵ[t+1] = sort(Distance[A[:,t],t])[floor(Int,Threshold*N)]
        elseif Criterion == "Unique"
            UniqueDist = sort(unique(Distance[A[:,t],t]))
            ϵ[t+1] = UniqueDist[floor(Int,Threshold*length(UniqueDist))]
        end
        # print("iteration",t,"ϵ",ϵ[t+1],"\n")
        W[:,t+1] = (Distance[:,t] .< ϵ[t+1])/sum(Distance[:,t] .< ϵ[t+1])
        #sigma = diagm(diag(cov(Particles[:,:,t])))
        sigma = scale*cov(Particles[A[:,t],:,t])
        for n = 1:N
            Particles[n,:,t+1] = LocalMH(Particles[A[n,t],:,t],sigma,ϵ[t+1],data,Method=Method,data=data)
            Distance[n,t+1] = dist(data,f(Particles[n,:,t+1],NoisyData=NoisyData,noise=noise,Method=Method,data=data))
        end
    end
    return (P=Particles,W=W,A=A,epsilon=ϵ,D=Distance)
end
