# try if the gradient function work properly
ξ = [rand(Uniform(0,10),4);rand(Normal(0,1),20)];


@time gradient(dist,ξ)

