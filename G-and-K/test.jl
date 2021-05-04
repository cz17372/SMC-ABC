include("main.jl")

R = BPS.BPS_SMC_ABC(1000,50,dat20,Threshold=0.8,δ=0.5,κ=0.5,K0=10)

using Plots


plot(R.EnergyBounceAccepted)
plot!(R.EnergyBounceProposed)

plot(R.BoundaryBounceAccepted)
plot!(R.BoundaryBounceProposed)

plot(10000*R.K)

plot(R.EnergyBounceAccepted ./  R.EnergyBounceProposed)

plot(R.BoundaryBounceAccepted ./ R.BoundaryBounceProposed)