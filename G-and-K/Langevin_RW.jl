using Distributions, Plots, StatsPlots, Random, Optim
using ForwardDiff: gradient,gradient!
using ForwardDiff: derivative
using ProgressMeter
using LinearAlgebra


function f(z,θ)
    return θ[1] + θ[2]*(1+0.8*((1-exp(-θ[3]*z))/(1+exp(-θ[3]*z))))*(1+z^2)^θ[4]*z
end 
function φ(ξ)
    θ = [ξ[1:2];[2.0];[ξ[3]]]
    z = ξ[4:end]
    return f.(z,Ref(θ))
end
function d(ξ;y)
    x = φ(ξ)
    return log(norm(y .- x))
end
function logPrior(ξ)
    θ = ξ[1:3]
    z = ξ[4:end]
    return sum(logpdf.(Uniform(0,10),θ)) + sum(logpdf.(Normal(0,1),z))
end

θ0 = [3.0,1.0,2.0,0.5];
Random.seed!(123);
z0 = rand(Normal(0,1),20);
y0 = φ([θ0;z0]);

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
        Acceptance_Prob_Vec[i]   = exp(min(0,new_particle_logprior-old_particle_logprior+log_backwardQ-log_forwadQ+log(dist(NewP[i,:]))))
    end
    return -mean(Jump_Distance_Vec.*Acceptance_Prob_Vec)
end

    
    


    