using Distributions, StatsPlots, Plots, KernelDensity, Measures

p = plot(layout=(3,4),size=(1600,1200),margin=10.0mm,framestyle=:box,xlabel=["" "" "" "" "" "" "" "" "a" "b" "g" "k"],ylabel="Density")