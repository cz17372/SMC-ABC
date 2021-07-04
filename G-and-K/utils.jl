module utils

using Plots, StatsPlots
theme(:wong2)


function plotinfo(R)
    Keys = keys(R.Information)

    for k in Keys
        println(k,"::",R.Information[k])
    end
end




end