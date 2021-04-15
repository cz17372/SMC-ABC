using Distributed, SharedArrays
addprocs(15)
@everywhere include("G-and-K/BPS/DistributedVersion/RW-SMC-ABC.jl")

@everywhere begin
    using Random
    Random.seed!(123)
    θstar = [3.0,1.0,2.0,0.5]
    zstar = rand(Normal(0,1),20)
    ystar = f.(zstar,θ=θstar)
end

begin
    @everywhere begin
        δ = 0.05
        K = 50
    end
    N = 10000; T= 200; NoData = 20; Threshold = 0.8
    U = SharedArray{Float64,3}(4+NoData,N,T+1)
    EPSILON = SharedArray{Float64,1}(T+1)
    DISTANCE = SharedArray{Float64,2}(N,T+1)
    WEIGHT   = SharedArray{Float64,2}(N,T+1)
    ANCESTOR = SharedArray{Int64,2}(N,T)
    for w in workers()
        @spawnat(w,WEIGHT)
        @spawnat(w,ANCESTOR)
        @spawnat(w,U)
        @spawnat(w,DISTANCE)
        @spawnat(w,EPSILON)
    end
    @time @sync @distributed for i = 1:N
        U[:,i,1] = [rand(Uniform(0,10),4);rand(Normal(0,1),20)]
        DISTANCE[i,1] = dist(U[:,i,1])
    end
        
    WEIGHT[:,1] .= 1/N
    EPSILON[1] = findmax(DISTANCE[:,1])[1]
    for t = 1:10
        print(t,"\n")
        for w in workers()
            @spawnat(w,t)
        end
        ANCESTOR[:,t] = vcat(fill.(1:N,rand(Multinomial(N,WEIGHT[:,t])))...);
        if length(unique(DISTANCE[ANCESTOR[:,t],t])) > Int(floor(0.4*N))
            EPSILON[t+1] = quantile(unique(DISTANCE[ANCESTOR[:,t],t]),Threshold)
        else
            EPSILON[t+1],_ = findmax(unique(DISTANCE[ANCESTOR[:,t],t]))
        end
        WEIGHT[:,t+1] = (DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])/sum(DISTANCE[ANCESTOR[:,t],t] .< EPSILON[t+1])
        @everywhere begin
            Σ = cov(U[:,findall(WEIGHT[:,t].>0),t],dims=2) + 1e-8*I
        end
        @sync @distributed for i = 1:N
            if DISTANCE[ANCESTOR[i,t],t] < EPSILON[t+1]
                U[:,i,t+1] = RWMH(K,U[:,ANCESTOR[i,t],t],ϵ=EPSILON[t+1],Σ=Σ,δ=δ)
                DISTANCE[i,t+1] = dist(U[:,i,t+1])
            end
        end
    end
end