using Distributions, Plots, StatsPlots, JLD2

@load "20data_delmoralsmc_1000Particles.jld2"

A = Results.U
density(A[1][1,:],label="")
for i= 2:30
    density!(A[i][1,:],label="")
end
current()

