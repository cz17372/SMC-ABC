using Distributions, Plots, StatsPlots, Random, StatsBase
theme(:ggplot2)
 # Implement LV model with variable number of latents

 function SimLVModel(θ,TerminalTime;x0=50,y0=100,dt=1.0)
    xvec = [x0]
    yvec = [y0]
    timevec = [0.0]
    nlat = 0 
    while true
        # Obtain the rate at which the next event occurs
        rate_vec = [θ[1]*xvec[end]*yvec[end], θ[2]*xvec[end], θ[3]*yvec[end], θ[4]*xvec[end]*yvec[end]]
        # Simulate waiting time to the next event
        waiting_time = rand(Exponential(1/sum(rate_vec)))
        nlat += 1
        push!(timevec,timevec[end]+waiting_time) # Get the next event time
        if timevec[end] > TerminalTime # If the next event time exceeds the Terminal Time, STOP. 
            break
        end
        event_index = findfirst(rand() .<= cumsum(rate_vec)/sum(rate_vec))
        #event_index = sample(1:4,Weights(rate_vec)) # Sample and simulate the next event
        nlat += 1
        if event_index == 1
            push!(xvec,xvec[end]+1)
            push!(yvec,yvec[end])
        elseif event_index == 2
            push!(xvec,xvec[end]-1)
            push!(yvec,yvec[end])
        elseif event_index == 3
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]+1)
        else
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]-1)
        end
    end
    obstime = collect(dt:dt:TerminalTime)
    obsx = zeros(length(obstime))
    obsy = zeros(length(obstime))
    for i = 1:length(obstime)
        ind = findlast(timevec .< obstime[i])
        obsx[i] = xvec[ind]
        obsy[i] = yvec[ind]
    end
    return (x=xvec[2:end],y=yvec[2:end],t=timevec[2:end-1],obsx=obsx,obsy=obsy,obstime=obstime,nlat=nlat)
end

nsamp = 1000
θ = [0.01,0.5,1.0,0.01]
NlatVec = zeros(nsamp)
for n = 1:nsamp
    data = SimLVModel(θ,50,x0=50,y0=100)
    NlatVec[n] = data.nlat
    GC.gc()
    println(n)
end