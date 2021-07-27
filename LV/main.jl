using Plots, StatsPlots, JLD2
theme(:wong2)
RW_10000Par1 = load("data/100data_RW10000Particles1_LV.jld2","Results");
RW_10000Par2 = load("data/100data_RW10000Particles2_LV.jld2","Results");

function PlotBatchRes(R,i;color=:grey,linewidth=0.1,newplot=true,size=(400,400),xlabel="",ylabel="",label="")
    if newplot
        density(R.U[1][i,:],color=color,linewidth=linewidth,size=size,xlabel=xlabel,ylabel=ylabel,label=label)
        for n = 2:length(R.U)
            density!(R.U[n][i,:],color=color,linewidth=linewidth,label="")
        end
        current()
    else
        density!(R.U[1][i,:],color=color,linewidth=linewidth,label=label)
        for n = 2:length(R.U)
            density!(R.U[n][i,:],color=color,linewidth=linewidth,label="")
        end
        current()
    end
end

theta = [0.4,0.005,0.05,0.001]
i = 4
PlotBatchRes(RW_10000Par1,i,color=:black);
PlotBatchRes(RW_10000Par2,i,color=:green,newplot=false);
vline!([(log(theta[i])+2)/3],label="",color=:red)

