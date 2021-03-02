R = NaiveSMCABC(10000,1500,20,Threshold=0.99,σ=0.3,λ=1.0)

RW_Unique = zeros(1500)
N_Unique  = zeros(1500)
for i = 1:1500
    RW_Unique[i] = length(unique(R2.DISTANCE[R2.ANCESTOR[:,i],i]))
    N_Unique[i]  = length(unique(R.DISTANCE[R.ANCESTOR[:,i],i]))
end
plot(RW_Unique,label="RW-SMC-ABCA",xlabel="Iteration",ylabel="Unique Starting Points")
plot!(N_Unique,label="Naive-SMC,ABC")

plot(log.(R.EPSILON)); plot!(log.(R2.EPSILON))

plot(R.ACCEPTANCE);plot!(R2.ACCEPTANCE)


plot(R.SIGMA);plot!(R2.SIGMA)

density(R2.XI[2,:,1000])

