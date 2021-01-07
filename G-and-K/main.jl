include("G-and-K/G-and-K.jl")

function PlotRes(StandardSMC,NewSMC,trueval,label,Iteration)
    """
    StandardSMC : The particles obtained from a standard SMC-ABC algorithm
    NewSMC      : The particles obtained from a new SMC-ABC algorithm
    trueval     : the ground truth of the parameter values
    Iteration   : the iteration of SMC we are plotting
    """

    density(StandardSMC,label="Standard SMC-ABC",xlab=label,color=:grey,linewidth=2.0,title="Iteration $(Iteration)"); 
    density!(NewSMC,label="New SMC-ABC",color=:green,linewidth=2.0);
    vline!([trueval],label="",color=:red,linewidth=2.0)
end




true_par = [3.0,1.0,2.0,0.5];
Random.seed!(123);
z,data = Generate_Data(20,par=true_par,NoisyData=true,noise=1.5);

"""
Generate an artificial data of size 20. Adding a Gaussian noise with mean 0 and standard deviation 0.5
to the g-and-k variables

Results were obtained by using both the standard SMC-ABC method and the new proposed SMC-ABC method.
For each simulation, 10,000 particles were sampled at each SMC step and a total of 1,000 SMC steps were 
performed. 

"""

New_SMCABC_RES = SMC(10000,1000,data,Criterion="ESS",Threshold=0.6,NoisyData=true,Method ="New",noise=1.5,scale=0.05);
Std_SMCABC_RES = SMC(10000,1000,data,Criterion="ESS",Threshold=0.6,NoisyData=true,Method="Standard",noise=1.5,scale=0.05);




t = 100; lab = ["a","b","g","k"]; param = 3;
p1 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t)
t = 500;
p2 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);

t = 800;
p3 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);

t = 1000;
p4 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);
plot(p1,p2,p3,p4,layout=(2,2),size=(500,500))


Random.seed!(123);
z,data = Generate_Data(20,par=true_par,NoisyData=false);
New_SMCABC_RES = SMC(10000,1000,data,Threshold=0.8,NoisyData=false,Method = "New",scale=0.02);
Std_SMCABC_RES = SMC(10000,1000,data,Threshold=0.8,NoisyData=false,Method="Standard",scale=0.02);

plot(log.(Std_SMCABC_RES.epsilon),label="Standard SMC-ABC")
plot!(log.(New_SMCABC_RES.epsilon),label="New SMC-ABC")

t = 100; lab = ["a","b","g","k"]; param = 1;
p1 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t)
t = 300;
p2 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);

t = 500;
p3 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);

t = 700;
p4 = PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t);
plot(p1,p2,p3,p4,layout=(2,2),size=(500,500))


Mat = New_SMCABC_RES.P[:,:,end]

NewMat = zeros(10000,20)
for i = 1:10000
    NewMat[i,:] = sort(Mat[i,5:end])
end

t = 
density(NewMat[:,t]);vline!([sort(z)[t]])

density(Mat[:,4])

function d(x;y)
    return sqrt(sum((sort(y).-sort(x)).^2))
end

@time ForwardDiff.gradient(x->d(x,y=data),Mat[1,5:end])

t = 850; lab = ["a","b","g","k"]; param = 4;
PlotRes(Std_SMCABC_RES.P[:,param,t],New_SMCABC_RES.P[:,param,t],true_par[param],lab[param],t)
