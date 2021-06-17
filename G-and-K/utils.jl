function plotposterior(Data,var;title="",label="",xlabel="",ylabel="",xlim=(0,1),color=:grey,linewidth=0.2,new=true,size=(600,600))
    U = Data.Theta
    no_replica = length(U)
    if new
        density(U[1][var,:],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewidth=linewidth,size=size,xlim=xlim)
        for m = 2:no_replica
            density!(U[m][var,:],label=label,color=color,linewidth=linewidth)
        end
        current()
    else
        density!(U[1][var,:],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewidth=linewidth,size=size,xlim=xlim)
        for m = 2:no_replica
            density!(U[m][var,:],label=label,color=color,linewidth=linewidth)
        end
        current()
    end
end


function plotK(Data;title="",label="",xlabel="",ylabel="",color=:grey,linewidth=0.2,new=true,size=(600,600))
    U = Data.K
    no_replica = length(U)
    if new
        plot(U[1],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewitdh=linewidth,size=size)
        for m = 2:no_replica
            plot!(U[m],label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewidth=linewidth)
        end
        current()
    else
        plot!(U[1],title=title,label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewitdh=linewidth,size=size)
        for m = 2:no_replica
            plot!(U[m],label=label,xlabel=xlabel,ylabel=ylabel,color=color,linewidth=linewidth)
        end
        current()
    end
end