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


end