include("G-and-K/G-and-K.jl")

true_par = [3.0,1.0,2.0,0.5];
Random.seed!(123);
data = Generate_Data(20,par=true_par,NoisyData=true,noise=0.5);
R = SMC(10000,1000,data,Threshold=0.6,NoisyData=true,noise=0.5,scale=0.1);
R2 = SMC(10000,1000,data,Threshold=0.6,NoisyData=true,Method="Standard",noise=0.5,scale=0.1)


t = 200
comb = [1,2]
scatter(R2.P[:,1,t],R2.P[:,2,t],markerstrokewidth=0.0,markersize=2.0,color=:grey,label="",xlim=(0,10),ylim=(0,10),size=(400,400))
scatter!(R.P[:,1,t],R.P[:,2,t],markerstrokewidth=0.0,markersize=2.0,color=:purple,label="",xlim=(0,10),ylim=(0,10),size=(400,400))

t = 200
density(R.P[:,1,t])
density!(R2.P[:,1,t])

plot(log.(R.epsilon))
plot!(log.(R2.epsilon))