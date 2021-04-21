using Base.Threads: @threads, @spawn
N = 10000;
T = 100;
NoData = 20;
Threshold = 0.8;
δ = 0.1;
K = 50;

U = zeros(4+NoData,N,T+1)
EPSILON = zeros(T+1)
DISTANCE = zeros(N,T+1)
WEIGHT = zeros(N,T+1)
ANCESTOR = zeros(Int,N,T)

@time begin
    @spawn for i = 1:N
        @async U[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
        @async DISTANCE[i,1] = dist(U[:,i,1])
    end
end

WEIGHT[:,1] .= 1/N
EPSILON[1] = findmax(DISTANCE[:,1])[1]

t = 2

ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
    EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
else
    EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
end
WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
index = findall(WEIGHT[:,t+1] .> 0.0)

@spawn for i = 1:length(index)
    U[:,index[i],t+1] = RWMH(K,U[:,ANCESTOR[index[i],t],t],EPSILON[t+1],Σ,δ)
    DISTANCE[index[i],t+1] = dist(U[:,index[i],t+1])
end

R = BPS_SMC_ABC(10000,10,20,Threshold=0.8,δ=0.1,refresh_rate = 0.6,K=50)

R = RW_SMC_ABC(10000,10,20,Threshold=0.8,δ=0.1,K=50)
Σ = cov(R.U[:,:,end],dims=2)
@time rand(Normal(0,1),24);
@time rand(MultivariateNormal(zeros(24),0.1^2*Σ));