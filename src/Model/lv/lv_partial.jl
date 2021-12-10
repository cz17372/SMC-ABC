module plvu
using Distributions

NoParam = 4
function ptheta(logθ)
    return sum(logpdf(Normal(μ,σ),logθ))
end

U(u) = sum(logpdf(Uniform(0,1),u))

function GenPar()
    return rand(Normal(μ,σ),NoParam)
end

function GenSeed(N)
    return rand(N)
end

function LV(θ,endtime,timestep,x0,y0)
    xvec = [x0]
    yvec = [y0]
    timevec = [0.0]
    nlat = 0
    while timevec[end] < endtime
        rate = θ[1]*xvec[end]*yvec[end] + θ[2]*xvec[end]+θ[3]*yvec[end]+θ[4]*xvec[end]*yvec[end]
        ti = rand(Exponential(1/rate))
        nlat +=1
        push!(timevec,timevec[end]+ti)
        probvec = [θ[1]*xvec[end]*yvec[end],θ[2]*xvec[end],θ[3]*yvec[end],θ[4]*xvec[end]*yvec[end]]/rate
        event = findfirst(rand() .< cumsum(probvec))
        nlat += 1
        if event == 1
            push!(xvec,xvec[end]+1)
            push!(yvec,yvec[end])
        elseif event == 2
            push!(xvec,xvec[end]-1)
            push!(yvec,yvec[end])
        elseif event == 3
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]+1)
        else
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]-1)
        end
    end
    outtime = collect(timestep:timestep:endtime)
    N = length(outtime)
    outx = zeros(N)
    outy = zeros(N)
    for i = 1:N
        index = findlast(outtime[i] .>= timevec)
        outx[i] = xvec[index]
        outy[i] = yvec[index]
    end
    return (time=outtime, data = [outx;outy], nlat = nlat)
end
function LV_Latents(θ,latents,endtime,timestep,x0,y0)
    xvec = [x0]
    yvec = [y0]
    timevec = [0.0]
    nlat = 0
    latentused = 0
    while timevec[end] < endtime
        rate = θ[1]*xvec[end]*yvec[end] + θ[2]*xvec[end]+θ[3]*yvec[end]+θ[4]*xvec[end]*yvec[end]
        if latentused < length(latents)
            latentused += 1
            ti = -1/rate*log(latents[latentused])
        else
            ti = rand(Exponential(1/rate))
        end
        nlat +=1
        push!(timevec,timevec[end]+ti)
        probvec = [θ[1]*xvec[end]*yvec[end],θ[2]*xvec[end],θ[3]*yvec[end],θ[4]*xvec[end]*yvec[end]]/rate
        if latentused < length(latents)
            latentused += 1
            event = findfirst(latents[latentused] .< cumsum(probvec))
        else
            event = findfirst(rand() .< cumsum(probvec))
        end
        nlat += 1
        if event == 1
            push!(xvec,xvec[end]+1)
            push!(yvec,yvec[end])
        elseif event == 2
            push!(xvec,xvec[end]-1)
            push!(yvec,yvec[end])
        elseif event == 3
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]+1)
        else
            push!(xvec,xvec[end])
            push!(yvec,yvec[end]-1)
        end
    end
    outtime = collect(timestep:timestep:endtime)
    N = length(outtime)
    outx = zeros(N)
    outy = zeros(N)
    for i = 1:N
        index = findlast(outtime[i] .>= timevec)
        outx[i] = xvec[index]
        outy[i] = yvec[index]
    end
    #println("No. latents used to simulate the data = ",nlat)
    return (time=outtime, data = [outx;outy])
end

function ϕ(u,logθ;endtime,timestep=1.0,x0=100,y0=100)
    data = LV_Latents(exp.(logθ),u,endtime,timestep,x0,y0)
    return data.data
end

function Ψ(u;endtime,timestep=1.0,x0=100,y0=100)
    logθ = σ*u[1:NoParam] .+ μ
    return ϕ(u[(NoParam+1):end],logθ,endtime=endtime,timestep=timestep,x0=x0,y0=y0)
end

end