function get_density(R,D,T;truepar,color=:red,linewidth=1.0,label="",xlabel="",ylabel="Density",new=true,size=(600,600))
    Index = findall(R.WEIGHT[:,T] .>0)
    if new
        density(R.U[T][D,Index],label=label,xlabel=xlabel,ylabel=ylabel,size=size,color=color,linewidth=linewidth)
        vline!([truepar],label="",color=:orange,linewidth=2.0)
    else
        density!(R.U[T][D,Index],label=label,color=color,linewidth=linewidth)
    end
end

function get_rw_pseudo_obs(R,T)
    g(x) = ϕ(x[5:end],θ=x[1:4])
    Index = findall(R.WEIGHT[:,T] .> 0)
    X = R.U[T][:,Index]
    D,N = size(X)
    out = zeros(D-4,N)
    for i = 1:N
        out[:,i] = g(X[:,i])
    end
    return out
end


function plot_data(data;linewidth=0.2)
    D,N = size(data)
    T = D ÷ 2
    plot(data[1:T,1],label="",color=:grey,linewidth=linewidth)
    plot!(data[T+1:end,1],label="",color=:grey,linewidth=linewidth)
    for n = 2:N
        plot!(data[1:T,n],label="",color=:grey,linewidth=linewidth)
        plot!(data[T+1:end,n],label="",color=:grey,linewidth=linewidth)
    end
    current()
end


function plot_obs(y;new=true,label=true,linewidth=1.0,size=(600,600),xlabel="",ylabel="")
    T = length(y) ÷ 2
    if new
        if label
            plot(y[1:T],xlabel=xlabel,ylabel=ylabel,size=size,linewidth=linewidth,color=:green,label="prey")
            plot!(y[T+1:end],label="predator",color=:red,linewidth=linewidth)
        else
            plot(y[1:T],xlabel=xlabel,ylabel=ylabel,size=size,linewidth=linewidth,color=:green,label="")
            plot!(y[T+1:end],label="",color=:red,linewidth=linewidth)
        end
    else
        if label
            plot!(y[1:T],linewidth=linewidth,color=:green,label="prey")
            plot!(y[T+1:end],label="predator",color=:red,linewidth=linewidth)
        else
            plot(y[1:T],linewidth=linewidth,color=:green,label="")
            plot!(y[T+1:end],label="",color=:red,linewidth=linewidth)
        end
    end
end


function get_std_psudo_obs(R,T)
    Index = findall(R.WEIGHT[:,T] .> 0)
    return R.X[T][:,Index]
end

