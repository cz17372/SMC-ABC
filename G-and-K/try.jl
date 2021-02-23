N = 10000; T = 100; NoData = 20; CoolingSchedule = 0.5; σ = 0.1; λ = 0.5; K = 10

XI       = zeros(4+NoData,N,T+1);
EPSILON  = zeros(T+1);
DISTANCE = zeros(N,T+1);
WEIGHT   = zeros(N,T+1);
ANCESTOR = zeros(Int,N,T);
for i = 1:N
    XI[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),NoData)]
    DISTANCE[i,1] = RW_Dist(XI[:,i,1])
end

error_deduction = zeros(T+1)
SIGMA    = zeros(T+1)
SIGMA[1] = σ
ACCEPTANCE = zeros(T)
WEIGHT[:,1] .= 1/N;
EPSILON[1]  = findmax(DISTANCE[:,1])[1]

error_deduction[1] = EPSILON[1]

t = 5

error_deduction[t+1] = min(error_deduction[t],CoolingSchedule*EPSILON[t])
EPSILON[t+1] = EPSILON[t] - error_deduction[t+1]
ANCESTOR[:,t] = collect(1:N)
for i = 1:N
    WEIGHT[i,t+1] = WEIGHT[i,t] * (RW_Dist(XI[:,ANCESTOR[i,t],t])<EPSILON[t+1])
end

while length(findall(WEIGHT[:,t+1].>0))<Int(0.3*N)
    error_deduction[t+1] = CoolingSchedule*error_deduction
    EPSILON[t+1] = EPSILON[t] - error_deduction[t+1]
    for i = 1:N
        WEIGHT[i,t+1] = WEIGHT[i,t] * (RW_Dist(XI[:,ANCESTOR[i,t],t])<EPSILON[t+1])
    end
end

WEIGHT[:,t+1] = WEIGHT[:,t+1]/sum(WEIGHT[:,t+1])
if length(findall(WEIGHT[:,t+1].>0))<Int(0.5*N)
    ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
    WEIGHT[:,t+1] .= 1/N
end
Σ = cov(XI[:,findall(WEIGHT[:,t].>0),t],dims=2)
accepted = zeros(N)
proposed = zeros(N)
Threads.@threads for i = 1:N
    print(i,"\n")
    global accepted,proposed
    XI[:,i,t+1],acc = RW_SMC_ABC_LocalMH(XI[:,ANCESTOR[i,t],t],EPSILON[t+1],Σ=Σ,σ=SIGMA[t],K=K)
    accepted[i] = acc
    proposed[i] = K
    DISTANCE[i,t+1] = RW_Dist(XI[:,i,t+1])
end

SIGMA[t+1] = exp(log(SIGMA[t]) + λ*(accepted-0.234))
ACCEPTANCE[t] = accepted


function SysResamp(W)
    N = length(W)
    u = rand(Uniform(0,1))
    W1 = W ./ sum(W)
    vvec = N*cumsum(W1)
    s = u
    m = 1
    A = zeros(N)
    for n = 1:N
        while vvec[m] < s
            m += 1
            s += 1
        end
        A[n] = m
    end
    return A 
end