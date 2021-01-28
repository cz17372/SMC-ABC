θ0 = [3.0,1.0,2.0,0.5];
Random.seed!(123);
z0 = rand(Normal(0,1),20);
y0 = φ([θ0;z0]);
R = SMC_Langevin(1000,100,y0, Threshold=0.8, s0=[0.3,0.3])

meanjump(x) = mean(x[findall(x .>0)])
findunique(x) = length(unique(x))

R2 = SMC_RW(1000,100,y0,Threshold=0.8,rang=[0.0,5.0])


function KDP(Langevin,RW,t,θ0,size)
    p1=density(Langevin.Particles[:,1,t],label="Langevin",xlabel="a",ylabel="density");
    density!(RW.Particles[:,1,t],label="RW");
    vline!([θ0[1]],label="True Value");

    p2=density(Langevin.Particles[:,2,t],label="Langevin",xlabel="b",ylabel="density");
    density!(RW.Particles[:,2,t],label="RW");
    vline!([θ0[2]],label="True Value");

    p3=density(Langevin.Particles[:,3,t],label="Langevin",xlabel="g",ylabel="density");
    density!(RW.Particles[:,3,t],label="RW");
    vline!([θ0[3]],label="True Value");

    p4=density(Langevin.Particles[:,4,t],label="Langevin",xlabel="K",ylabel="density");
    density!(RW.Particles[:,4,t],label="RW");
    vline!([θ0[4]],label="True Value");
    plot(p1,p2,p3,p4,layout=(2,2),size=size)
end

function PlotResult(Langevin,RW,size)
    # Plot the epsilon values - iteration
    p1 = plot(Langevin.Epsilon,xlabel="Iteration",ylabel="Epsilon",label="Langevin")
    plot!(RW.Epsilon,label="RW")

    # Plot the averate jump distance at each iteration
    p2 = plot(mapslices(meanjump,Langevin.JumpDistance,dims=1)[1,:],label="Langevin",xlabel="Iteration",ylabel="Expected Jump Distance")
    plot!(mapslices(meanjump,RW.JumpDistance,dims=1)[1,:],label="RW")

    

    # Plot the acceptance proportion at each iteration
    p3 = plot(Langevin.AcceptancePortion,label="Langevin",xlabel="Iteration",ylabel="Acceptance Proportion",ylim=(0,1))
    plot!(RW.AcceptancePortion,label="RW")

    # Plot the number of unique particles at each iteration
    p4 = plot(mapslices(findunique,Langevin.Distance,dims=1)[1,:],label="Langevin",xlabel="Iteration",ylabel="No. Unique Particles")
    plot!(mapslices(findunique,RW.Distance,dims=1)[1,:],label="RW")

    plot(p1,p2,p3,p4,layout=(2,2),size=size)
end



KDP(R,R2,100,θ0,(800,800))
savefig("Experiment3_KDE.pdf")

PlotResult(R,R2,(800,800))
savefig("Experiment3_Results.pdf")

boxplot(R.Particles[:,4,end],label="Langevin")
boxplot!(R2.Particles[:,4,end],label="RW")

plot(log.(abs.(R.OptimalScale[:,1])))