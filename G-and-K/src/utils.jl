module utils

using Distributions, Plots, StatsPlots
using PyCall
using Random
using LinearAlgebra

py"""
import numpy as np
from scipy.stats import chi2
from scipy.special import gammaln

def fminESS(p, alpha=.05, eps=.05, ess=None):
    crit = chi2.ppf(1 - alpha, p)
    foo = 2. / p

    if ess is None:
        logminESS = foo * np.log(2.) + np.log(np.pi) - foo * np.log(p) -\
            foo * gammaln(p / 2.) - 2. * np.log(eps) + np.log(crit)
        return np.round(np.exp(logminESS))
    else:
        if isinstance(ess, str):
            raise ValueError("Only numeric entry allowed for ess")
        logEPS = .5 * foo * np.log(2.) + .5 * np.log(np.pi) -\
            .5 * foo * np.log(p) - .5 * foo * gammaln(p / 2.) -\
            .5 * np.log(ess) + .5 * np.log(crit)
        return np.exp(logEPS)


def multiESS(X, b='sqroot', Noffsets=10, Nb=None):
    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective "
            "sample size.")

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'.")
        if b != 'less' and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B).")
    else:
        if not 1. < b < (n / 2):
            raise ValueError(
                "The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):

    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(map(int, np.round(np.exp(
            np.linspace(np.log(b_min), np.log(b_max), Nb)))))

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(list(set(map(int, np.round(
        np.linspace(0, n - np.dot(a, b), Noffsets))))))

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    mESS = n * (detLambda / np.linalg.det(Sigma)) ** (1. / p)

    return mESS
"""
ESS(x) = py"multiESS"(x) 


f(z;θ) = θ[1] + θ[2]*(1+0.8*(1-exp(-θ[3]*z))/(1+exp(-θ[3]*z)))*(1+z^2)^θ[4]*z;

function DataGenerator(θ,N,seed)
    Random.seed!(seed)
    z = rand(Normal(0,1),N)
    return f.(z,θ=θ)
end



# Diagnostic Plots for RW-ABC-SMC
function densityplot(R,dim,T;color=:red,size=(500,500),xlabel="",ylabel="density",p=nothing,label="",linewidth=1.0)
    Index = findall(R.WEIGHT[:,T] .> 0)
    X     = R.U[T][:,Index]
    X[1:4,:] = 10*cdf(Normal(0,1),X[1:4,:])
    if p === nothing
        density(X[dim,:],color=color,size=size,xlabel=xlabel,ylabel=ylabel,label=label,linewidth=linewidth)
    else
        density!(p,X[dim,:],color=color,label=label,linewidth=linewidth)
    end
end

function MCStepPerESS(R)
    ESS = R.ESS[end]
    T = length(R.U)
    Tot = 0
    for t = 1:(T-1)
        Tot += sum(R.WEIGHT[:,t+1] .> 0)*R.K[t]
    end
    return Tot/ESS
end

function transferTheta(R,T)
    Index = findall(R.WEIGHT[:,T] .> 0)
    X     = transpose(10*cdf(Normal(0,1),R.U[T][1:4,Index]))
    return X
end

NC(R) = sum(log.(mapslices(x->sum(x.>0),R.WEIGHT,dims=1) / 5000))

end