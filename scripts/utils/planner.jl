reversedims(A::Array) = permutedims(A, ndims(A):-1:1) # @generated probably the "right" way

for MAType in (Float32MultiArray, Float64MultiArray, Int32MultiArray, Int64MultiArray)
    @eval function Base.convert(::Type{$MAType}, A::Array{<:Real})
        Arev = reversedims(A)
        msg = $MAType()
        msg.layout.dim = [MultiArrayDimension("",d,s) for (d,s) in zip(size(A), reverse(cumprod([size(Arev)...])))]
        msg.data = Arev[:]
        msg
    end

    @eval function Base.convert(::Type{Array{T}}, msg::$MAType) where {T}
        if isempty(msg.data)
            T[]
        else
            reversedims(reshape(msg.data, (Int(d.size) for d in msg.layout.dim[end:-1:1])...))
        end
    end
end

mutable struct PredictionOutputHolder
    stale::Bool
    y
    z
    r
    b
end

function prediction_callback(msg::PredictionOutput, pred_container::PredictionOutputHolder)
    pred_container.stale = false
    pred_container.y = Array{Float32}(msg.y)
    pred_container.z = Array{Float32}(msg.z)
    pred_container.r = Array{Float32}(msg.r)
    pred_container.b = Array{Float32}(msg.b)
end

const pred_container = PredictionOutputHolder(true,nothing,nothing,nothing,nothing)
const publisher = Publisher{PredictionInput}("prediction_input", queue_size=10)
const subscriber = Subscriber{PredictionOutput}("prediction_output", prediction_callback, (pred_container,), queue_size=10)

function predict!(car1, car2, extras, traj_lengths, car1_future, sample_ct,
                  output::PredictionOutputHolder=pred_container, timeout=1000)
    output.stale = true
    publish(publisher, PredictionInput(Float32MultiArray(car1),
                                       Float32MultiArray(car2),
                                       Float32MultiArray(extras),
                                       Int32MultiArray(traj_lengths),
                                       Float32MultiArray(car1_future),
                                       Float32MultiArray(Float32[]),
                                       Float32MultiArray(Float32[]),
                                       Int32(sample_ct)))
    for i in 1:timeout
        rossleep(Duration(0.001))
        output.stale || break
    end
    output
end

function predict!(car1, car2, extras, traj_lengths, car1_future_x, car1_future_y, sample_ct,
                  output::PredictionOutputHolder=pred_container, timeout=1000)
    output.stale = true
    publish(publisher, PredictionInput(Float32MultiArray(car1),
                                       Float32MultiArray(car2),
                                       Float32MultiArray(extras),
                                       Int32MultiArray(traj_lengths),
                                       Float32MultiArray(Float32[]),
                                       Float32MultiArray(car1_future_x),
                                       Float32MultiArray(car1_future_y),
                                       Int32(sample_ct)))
    for i in 1:timeout
        rossleep(Duration(0.001))
        output.stale || break
    end
    output
end

######
# const y_top = -1.83284f0
# const y_bot = -6.09485f0
const y_top = -1.5f0
const y_bot = -6.5f0
const xdd_choices = Float32[0, -3, 4, -6]
const y_targets = [y_top, y_bot]
const tibvp = SteeringBVP(TripleIntegratorDynamics(1, Float32), TimePlusQuadraticControl(SMatrix{1,1}(Float32(1e-3))))

trapz(x::Vector, dt, x0=0) = cumsum(vcat(0, 0.5*dt*(x[1:end-1] + x[2:end]))) + x0
trapz(x::Matrix, dt, x0=0) = cumsum(hcat(zeros(size(x,1)), 0.5*dt*(x[:,1:end-1] + x[:,2:end])), 2) .+ x0
vel(xd, yd) = sqrt.(xd.^2 + yd.^2)
acc(xd, xdd, yd, ydd, v=vel(xd,yd)) = (xd.*xdd + yd.*ydd)./v
curv(xd, xdd, yd, ydd, v=vel(xd,yd)) = (xd.*ydd - xdd.*yd)./v.^3

@inbounds @views function all_future_xy(xdd_choices, y_targets, state0, tibvp,
                                        a0 = [(1,1)], ply=4, t_hold=3, dt=Float32(0.1);
                                        des_length=length(a0)+ply*t_hold)
    Nx = length(xdd_choices)
    Ny = length(y_targets)
    X = zeros(Float32, Nx^ply, 1 + des_length, 3)
    Y = zeros(Float32, Ny^ply, 1 + des_length, 3)
    if a0 isa Vector{Tuple{Int,Int}}
        X[:,1,:] .= state0[1:2:end]'
        Y[:,1,:] .= state0[2:2:end]'
        t0 = 1 + length(a0)
        for (t, (xi, yi)) in enumerate(a0)
            X[:,t+1,3] .= xdd_choices[xi]
            q0 = SVector(Y[1,t,1],
                         Y[1,t,2],
                         Y[1,t,3])
            qf = SVector(y_targets[yi], 0f0, 0f0)
            c, u = tibvp(q0, qf, 100f0)
            if dt > duration(u)
                q = qf
            else
                q = u.x(q0, u.y, u.t, dt)
            end
            Y[:,t+1,1] .= q[1]
            Y[:,t+1,2] .= q[2]
            Y[:,t+1,3] .= q[3]
        end
    elseif a0 isa Tuple
        Ẍ0, Ÿ0 = a0
        X[:,1,:] .= state0[1:2:end]'
        Y[:,1,:] .= state0[2:2:end]'
        X[:,2:1+length(Ẍ0),3] .= Ẍ0'
        Y[:,2:1+length(Ÿ0),3] .= Ÿ0'
        t0 = 1 + length(Ÿ0)
        cumsum!(Y[:,2:t0,2], Y[:,1:t0-1,3]*dt, 2)    # X integrated below (joint with the case above)
        Y[:,2:t0,2] .+= Y[:,1,2]
        cumsum!(Y[:,2:t0,1], (Y[:,1:t0-1,2] .+ Y[:,2:t0,2])*dt./2, 2)
        Y[:,2:t0,1] .+= Y[:,1,1]
    else
        error("Incompatible a0 type.")
    end

    # xdd, xd, x
    for (r, ci) in enumerate(CartesianRange(Tuple(Nx for i in 1:ply)))
        for (p, v) in enumerate(reverse(ci.I))    # reverse so that later indices (actions for later timesteps) iterate fastest
            for t in 1:t_hold
                X[r,t0+t_hold*(p-1)+t,3] = xdd_choices[v]
            end
        end
    end
    for t in (t0+ply*t_hold+1):1+des_length
        X[:,t,3] = X[:,t-1,3]
    end
    cumsum!(X[:,2:end,2], X[:,1:end-1,3]*dt, 2)
    X[:,2:end,2] .+= X[:,1,2]
    cumsum!(X[:,2:end,1], (X[:,1:end-1,2] .+ X[:,2:end,2])*dt./2, 2)
    X[:,2:end,1] .+= X[:,1,1]

    # ydd, yd, y
    for (r, ci) in enumerate(CartesianRange(Tuple(Ny for i in 1:ply)))
        for (p, v) in enumerate(reverse(ci.I))    # reverse so that later indices (actions for later timesteps) iterate fastest
            curr_step = t0 + (p-1)*t_hold
            q0 = SVector(Y[r,curr_step,1],
                         Y[r,curr_step,2],
                         Y[r,curr_step,3])
            qf = SVector(y_targets[v], 0f0, 0f0)
            c, u = tibvp(q0, qf, 100f0)
            for t in 1:t_hold
                if dt*t > duration(u)
                    q = qf
                else
                    q = u.x(q0, u.y, u.t, t*dt)
                end
                Y[r,curr_step+t,1] = q[1]
                Y[r,curr_step+t,2] = q[2]
                Y[r,curr_step+t,3] = q[3]
            end
        end
    end
    for t in (t0+ply*t_hold+1):1+des_length
        Y[:,t,3] = Y[:,t-1,3]
        Y[:,t,2] = Y[:,t-1,2] + Y[:,t-1,3]*dt
        Y[:,t,1] = Y[:,t-1,1] + Y[:,t-1,2]*dt
    end
    X, Y
end

function propagate(xdd_choices, y_targets, car1_curr_state, tibvp, action_sequence)
    pfX, pfY = all_future_xy(xdd_choices, y_targets, car1_curr_state, tibvp, action_sequence, 0)
    [SVector{6,Float32}(pfX[1,i,1], pfY[1,i,1],
                        pfX[1,i,2], pfY[1,i,2],
                        pfX[1,i,3], pfY[1,i,3]) for i in 1:size(pfX, 2)]
end
