module utils
using Distributions
function RWSMC_CompCost(R)
    T = length(R.EPSILON) - 1
    Cost = 0
    for n = 1:T
        Cost += sum(R.WEIGHT[:,n+1] .> 0) * R.K[n]
    end
    return Cost/sum(R.WEIGHT[:,end] .> 0)
end

function gkn_getsamp(R)
    Index = findall(R.WEIGHT[:,end] .> 0)
    X     = 10*cdf(Normal(0,1),R.U[end][1:4,Index])
    return X
end

function smc_getsamp(R)
    Index = findall(R.WEIGHT[:,end] .> 0)
    X = R.U[end][:,Index]
    return X
end

end