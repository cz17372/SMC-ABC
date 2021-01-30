using Distributions, Plots, StatsPlots, Random, Optim
using Flux: gradient
using ForwardDiff: derivative
using ProgressMeter
using LinearAlgebra
theme(:mute)


function f(z,θ)
    return θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
end
function φ(ξ)
    θ = ξ[1:4]
    z = ξ[5:end]
    return f.(z,Ref(θ))
end

logPrior(ξ) = sum(logpdf.(Uniform(0,10),ξ[1:4]))+sum(logpdf.(Normal(0,1),ξ[5:end]))


function ESJD_Langevin(sigma;P0,U,Σ,epsilon,GradP,grad,dist)
    #=
    sigma     : vector of the scale factors, for Langevin-style update, this is [σ1,σ2]
    P0        : the matrix of particles from previous iteration
    U         : the random seeds (i.e. random Normals) used to make proposals in the current iteration
    Σ         : the variance-covariance matrix used to generate the random matrix U 
    epsilon   : the threshold distance used in the current iteration
    GradP     : the gradient matrix of the log-distance function at each particle value ξ_{n-1}^i for i = 1,2,...,N
    grad      : the function need to calculate the gradients at the proposals
    dist      : distance metric used in the calcualtion
    =#

    #=
    The function returns the empirical Expected-Squared-Jump-Distance (ESJD) corresponding to the scale
    factors sigma = [σ1,σ2]
    =#
    
    # Find the dimensions of P0, N: no. rows; C: number of columns
    N,C = size(P0)

    # Proposal the new particles by the Langevin dynamics ξ' = ξ_{n-1} - sigma[1]*∇d(ξ_{n-1}) + sigma[2]*u
    # NewP: the matrix of proposals, the i-th row of NewP corresponds to the proposal based on ξ_{n-1}^i
    NewP = P0 .- sigma[1]*GradP .+ sigma[2]*U
    
    Acceptance_Prob_Vec = zeros(N)
    Jump_Distance_Vec   = (mapslices(norm,P0 .- NewP,dims=2)[:,1]).^2
    
    # Calcualte the gradients at the proposals for calculating the backward proposal density q(ξ_{n-1}|ξ')
    Proposal_Grad = mapslices(grad,NewP,dims=2)

    for i = 1:N
        forward_proposal_density = MultivariateNormal(P0[i,:] .- sigma[1]*GradP[i,:],sigma[2]^2*Σ)
        backward_proposal_density= MultivariateNormal(NewP[i,:] .- sigma[1]*Proposal_Grad[i,:],sigma[2]^2*Σ)
        old_particle_logprior    = logPrior(P0[i,:])
        new_particle_logprior    = logPrior(NewP[i,:])
        log_forwardQ             = logpdf(forward_proposal_density,NewP[i,:])
        log_backwardQ            = logpdf(backward_proposal_density,P0[i,:])
        Acceptance_Prob_Vec[i]   = exp(min(0,new_particle_logprior-old_particle_logprior+log_backwardQ-log_forwardQ+log(dist(NewP[i,:])<epsilon)))
    end
    return mean(Jump_Distance_Vec.*Acceptance_Prob_Vec)
end

function MHUpdate_Langevin(sigma;P0,U,Σ,epsilon,GradP,grad,dist)
    #=
    This function takes the same inputs as the function ESJD_Langevin. The differences is that the output of this function will be the results of Langevin-style update based on each particle in P0, using the optimal choice of sigma (i.e., sigma value that maximise ESJD)
    =#
    
    # Find the dimensions of P0, N: no. rows; C: number of columns
    N,C = size(P0)

    # Proposal the new particles by the Langevin dynamics ξ' = ξ_{n-1} - sigma[1]*∇d(ξ_{n-1}) + sigma[2]*u
    # NewP: the matrix of proposals, the i-th row of NewP corresponds to the proposal based on ξ_{n-1}^i
    NewP = P0 .- sigma[1]*GradP .+ sigma[2]*U
    
    Acceptance_Prob_Vec = zeros(N)
    
    Acceptance_Ind_Vec = ones(Int64,N)
    # Calcualte the gradients at the proposals for calculating the backward proposal density q(ξ_{n-1}|ξ')
    Proposal_Grad = mapslices(grad,NewP,dims=2)

    for i = 1:N
        # obtain the forward proposal and backward proposal
        forward_proposal_density = MultivariateNormal(P0[i,:] .- sigma[1]*GradP[i,:],sigma[2]^2*Σ)
        backward_proposal_density= MultivariateNormal(NewP[i,:] .- sigma[1]*Proposal_Grad[i,:],sigma[2]^2*Σ)
        # calculate the priors
        old_particle_logprior    = logPrior(P0[i,:])
        new_particle_logprior    = logPrior(NewP[i,:])
        # calculate the proposal densities
        log_forwardQ             = logpdf(forward_proposal_density,NewP[i,:])
        log_backwardQ            = logpdf(backward_proposal_density,P0[i,:])
        # calculate the acceptance probabilities for the proposal
        Acceptance_Prob_Vec[i]   = exp(min(0,new_particle_logprior-old_particle_logprior+log_backwardQ-log_forwardQ+log(dist(NewP[i,:])<epsilon)))

        u = rand(Uniform(0,1))

        if u >= Acceptance_Prob_Vec[i]
            # Upon rejection, set NewP[i,:] = P0[i,:]
            NewP[i,:] = P0[i,:]
            Acceptance_Ind_Vec[i] = 0
        end
    end
    return (NewParticles=NewP,Average_Accept_Rate = mean(Acceptance_Prob_Vec),Decision = Acceptance_Ind_Vec)
end

function LocalMH_Langevin(P0,epsilon,s0;Σ,GradP,grad,dist)
    #=
    LocalMH_Langevin performs the local Metropolis-Hastings exploration. It uses particles from the previous SMC step as starting point and explore the space once using Langevin-style proposals. 
    The function will return the updated particles 
    =#
    N,C = size(P0)
    U = zeros(N,C)
    for i = 1:N
        U[i,:] = rand(MultivariateNormal(zeros(C),Σ))
    end

    object(x) = -ESJD_Langevin(x,P0=P0,U=U,Σ=Σ,epsilon=epsilon,GradP=GradP,grad=grad,dist=dist)
    
    Opt = optimize(object,s0)


    opt_simga = Opt.minimizer
    NewP,AcceptRate,Decision=MHUpdate_Langevin(opt_simga,P0=P0,U=U,Σ=Σ,epsilon=epsilon,GradP=GradP,grad=grad,dist=dist)
    return (Opt_Sigma=opt_simga,NewP=NewP,AcceptRate=AcceptRate,Decision=Decision)
end

function SMC_Langevin(N,T,y;Threshold,s0)
    C       = length(y)
    P       = zeros(N,C+4,T+1);
    GradP   = zeros(N,C+4,T);
    W       = zeros(N,T+1);
    A       = zeros(Int64,N,T);


    D       = zeros(N,T+1);
    JumpD   = zeros(N,T);

    ϵ       = zeros(T+1);
    α       = zeros(T);
    SMat    = zeros(T+1,2);
    SMat[1,:] = s0;



    d(ξ)    = norm(φ(ξ) .- y);
    grad(ξ) = gradient(d,ξ);
    
    # The particles at the 0-th step of the SMC are sampled from 
    # the prior
    for i = 1:N
        P[i,:,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),C)]
        D[i,1]   = d(P[i,:,1])
    end

    ϵ[1] = findmax(D[:,1])[1]
    W[:,1] .= 1/N;

    @showprogress 1 "Computing..." for t = 1:T
        # Sample the ancestor for each particle
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...);

        # Search for the next ϵ value
        ϵ[t+1] = quantile(D[:,t],Threshold);
        
        W[:,t+1] = (D[A[:,t],t] .< ϵ[t+1])/sum(D[A[:,t],t] .< ϵ[t+1])

        Σ = cov(P[A[:,t],:,t]);
        #Σ = I;
        GradP[:,:,t] = mapslices(grad,P[:,:,t],dims=2)

        opt_s,newp,acc,_ = LocalMH_Langevin(P[A[:,t],:,t],ϵ[t+1],SMat[t,:],Σ=Σ,GradP = GradP[A[:,t],:,t],grad=grad,dist=d)

        SMat[t+1,:] = opt_s;
        P[:,:,t+1]  = newp;
        α[t]        = acc;
        D[:,t+1]    = mapslices(d,newp,dims=2);
        JumpD[:,t]  = mapslices(norm, P[A[:,t],:,t] .- newp, dims=2)
    end

    return (Particles = P, Ancestors = A, Weights = W, Epsilon = ϵ, JumpDistance = JumpD, Distance = D, AcceptancePortion = α, OptimalScale=SMat)
end



function ESJD_RW(sigma;P0,U,Σ,epsilon,dist)
    N,C = size(P0)
    NewP = P0 .+ sigma*U
    Acceptance_Prob_Vec = zeros(N)
    Jump_Distance_Vec = (mapslices(norm,P0 .- NewP,dims=2)[:,1]).^2

    for i = 1:N
        old_particle_logprior    = logPrior(P0[i,:])
        new_particle_logprior    = logPrior(NewP[i,:])
        Acceptance_Prob_Vec[i]   = exp(min(0,new_particle_logprior-old_particle_logprior+log(dist(NewP[i,:])<epsilon)))
    end
    return mean(Jump_Distance_Vec.*Acceptance_Prob_Vec)
end

function MHUpdate_RW(sigma;P0,U,Σ,epsilon,dist)
    N,C = size(P0)
    NewP = P0 .+ sigma * U
    Acceptance_Prob_Vec = zeros(N)
    Acceptance_Ind_Vec  = ones(Int64,N)
    for i = 1:N
        old_particle_logprior    = logPrior(P0[i,:])
        new_particle_logprior    = logPrior(NewP[i,:])
        Acceptance_Prob_Vec[i]   = exp(min(0,new_particle_logprior-old_particle_logprior+log(dist(NewP[i,:])<epsilon)))
        u = rand(Uniform(0,1))

        if u >= Acceptance_Prob_Vec[i]
            NewP[i,:] = P0[i,:]
            Acceptance_Ind_Vec[i] = 0
        end
    end
    return (NewParticles=NewP,Average_Accept_Rate = mean(Acceptance_Prob_Vec),Decision = Acceptance_Ind_Vec)
end

function LocalMH_RW(P0,epsilon,rang;Σ,dist)
    N,C = size(P0)
    U = zeros(N,C)
    for i = 1:N
        U[i,:] = rand(MultivariateNormal(zeros(C),Σ))
    end
    object(x) = -ESJD_RW(x,P0=P0,U=U,Σ=Σ,epsilon=epsilon,dist=dist)
    Opt = optimize(object,rang[1],rang[2])
    opt_sigma = Opt.minimizer
    NewP, AcceptRate, Decision = MHUpdate_RW(opt_sigma,P0=P0,U=U,Σ=Σ,epsilon=epsilon,dist=dist)
    return (Opt_Sigma=opt_sigma,NewP=NewP,AcceptRate=AcceptRate,Decision=Decision)
end

function SMC_RW(N,T,y;Threshold,rang)
    C       = length(y)
    P       = zeros(N,C+4,T+1);
    W       = zeros(N,T+1);
    A       = zeros(Int64,N,T);


    D       = zeros(N,T+1);
    JumpD   = zeros(N,T);
    ϵ       = zeros(T+1);
    α       = zeros(T);
    SVec    = zeros(T);
    d(ξ)    = norm(φ(ξ) .- y);

    for i = 1:N
        P[i,:,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),C)]
        D[i,1]   = d(P[i,:,1])
    end

    ϵ[1] = findmax(D[:,1])[1]
    W[:,1] .= 1/N;

    @showprogress 1 "Computing..." for t = 1:T
        A[:,t] = vcat(fill.(1:N,rand(Multinomial(N,W[:,t])))...);
        if length(unique(D[:,t]))<500
            ϵ[t+1] = ϵ[t]
        else
            ϵ[t+1] = sort(unique(D[:,t]))[500]
        end
        W[:,t+1] = (D[A[:,t],t] .< ϵ[t+1])/sum(D[A[:,t],t] .< ϵ[t+1])
        Σ = cov(P[A[:,t],:,t]);
        #Σ = I;
        opt_s,newp,acc,_ = LocalMH_RW(P[A[:,t],:,t],ϵ[t+1],rang,Σ=Σ,dist=d)
        SVec[t] = opt_s
        P[:,:,t+1] = newp
        α[t] = acc;
        D[:,t+1] = mapslices(d,newp,dims=2);
        JumpD[:,t]  = mapslices(norm, P[A[:,t],:,t] .- newp, dims=2)
    end
    return (Particles = P, Ancestors = A, Weights = W, Epsilon = ϵ, JumpDistance = JumpD, Distance = D, AcceptancePortion = α, OptimalScale=SVec)
end



