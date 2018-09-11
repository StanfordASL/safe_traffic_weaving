module TrafficWeavingPlanner

const robot = "/robot"
const human = "/human"

using PyCall
using RobotOS
using StaticArrays
using Rotations
using Interpolations

using TrafficWeavingPlanner
using DifferentialDynamicsModels
using LinearDynamicsModels

@pyimport rospy
@pyimport tf2_ros
@rosimport geometry_msgs.msg: Transform, TransformStamped, PoseStamped, TwistStamped, AccelStamped
@rosimport std_msgs.msg: Float32MultiArray, Float64MultiArray,
                         Int32MultiArray, Int64MultiArray,
                         MultiArrayLayout, MultiArrayDimension,
                         Bool
@rosimport safe_traffic_weaving.msg: Float32MultiArrayStamped, PredictionInput, PredictionOutput
rostypegen()
import geometry_msgs.msg: Transform, TransformStamped, PoseStamped, TwistStamped, AccelStamped
import std_msgs.msg: Float32MultiArray, Float64MultiArray,
                     Int32MultiArray, Int64MultiArray,
                     MultiArrayLayout, MultiArrayDimension,
                     BoolMsg
import safe_traffic_weaving.msg: Float32MultiArrayStamped, PredictionInput, PredictionOutput

# const tfBuffer = tf2_ros.Buffer()
# const listener = tf2_ros.TransformListener(tfBuffer)
# lane_100_py_tf = tfBuffer[:lookup_transform]("world", "lane_100", rospy.Time(), rospy.Duration(1.0))
# lane_0_py_tf = tfBuffer[:lookup_transform]("world", "lane_0", rospy.Time(), rospy.Duration(1.0))

const Timeseries2D = Interpolations.GriddedInterpolation{SVector{2,Float64},1,SVector{2,Float32},Interpolations.Gridded{Interpolations.Linear},Tuple{Array{Float64,1}},0}
Timeseries2D() = interpolate((Float64[],), SVector{2,Float32}[], Gridded(Linear()))
truncate!(v::Vector) = resize!(v, 0)
truncate!(ts::Timeseries2D) = (foreach(truncate!, ts.knots); truncate!(ts.coefs))
Base.push!(ts::Timeseries2D, t, x) = (push!(ts.knots[1], t); push!(ts.coefs, x))
Base.push!(ts::Timeseries2D, tx::Tuple) = push!(ts, tx[1], tx[2])

# struct Timeseries2D
#     t::Vector{Float64}
#     xy::Vector{SVector{2,Float32}}
# end
# Timeseries2D() = Timeseries2D(Float64[], SVector{2,Float32}[])
# truncate!(v::Vector) = resize!(v, 0)
# truncate!(ts::Timeseries2D) = (truncate!(ts.t); truncate!(ts.xy))

struct DoubleIntegratorTimeseries2D
    pos::Timeseries2D
    vel::Timeseries2D
    acc::Timeseries2D
end
DoubleIntegratorTimeseries2D() = DoubleIntegratorTimeseries2D(Timeseries2D(), Timeseries2D(), Timeseries2D())
truncate!(dits::DoubleIntegratorTimeseries2D) = (truncate!(dits.pos); truncate!(dits.vel); truncate!(dits.acc))
Base.getindex(dits::DoubleIntegratorTimeseries2D, t) = [dits.pos[t]; dits.vel[t]; dits.acc[t]]
function last_common_time(dits::DoubleIntegratorTimeseries2D)
    if length(dits.pos.knots[1]) > 0 && length(dits.vel.knots[1]) > 0 && length(dits.acc.knots[1]) > 0
        min(dits.pos.knots[1][end], dits.vel.knots[1][end], dits.acc.knots[1][end])
    else
        -1.0
    end
end
function first_common_time(dits::DoubleIntegratorTimeseries2D)
    if length(dits.pos.knots[1]) > 0 && length(dits.vel.knots[1]) > 0 && length(dits.acc.knots[1]) > 0
        max(dits.pos.knots[1][1], dits.vel.knots[1][1], dits.acc.knots[1][1])
    else
        -1.0
    end
end

mutable struct PlannerState
    initialized::Bool
    terminate::Bool
    start_time::Float64
    computed_up_to_t::Int
    human::Vector{SVector{6,Float32}}     # car2
    robot::Vector{SVector{6,Float32}}     # car1
    human_buffer::DoubleIntegratorTimeseries2D
    robot_buffer::DoubleIntegratorTimeseries2D
    # robot_actions::Vector{Tuple{Int,Int}}
    robot_des::Vector{SVector{6,Float32}} # identical to robot up to current time, but includes future actions
    robot_l0::Int
    robot_lf::Int
    # robot_a0::Vector{Tuple{Int,Int}}    # current queue of actions to be implemented
    # robot_a0_t0::Int                    # timestep of current action
end

mutable struct HumanCarStates
    t::Vector{Float32}
    X::Vector{SVector{6,Float32}}
end

const planner_state = PlannerState(false, false, 0.0, 0,
                                   SVector{6,Float32}[],
                                   SVector{6,Float32}[],
                                   DoubleIntegratorTimeseries2D(),
                                   DoubleIntegratorTimeseries2D(),
                                   # Tuple{Int,Int}[],
                                   SVector{6,Float32}[], 0, 0)

get_xy(msg::PoseStamped) = SVector{2,Float32}(msg.pose.position.x, msg.pose.position.y)
get_xy(msg::TwistStamped) = SVector{2,Float32}(msg.twist.linear.x, msg.twist.linear.y)
get_xy(msg::AccelStamped) = SVector{2,Float32}(msg.accel.linear.x, msg.accel.linear.y)
function append_callback(msg, ts)
    xy = get_xy(msg)
    !isnan(xy[1]) && !isnan(xy[2]) && push!(ts, Float64(msg.header.stamp), xy)
end

# human_pos_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.pose.position.x,
#                                                                                                        msg.pose.position.y))
# human_vel_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.twist.linear.x,
#                                                                                                        msg.twist.linear.y))
# human_acc_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.accel.linear.x,
#                                                                                                        msg.accel.linear.y))
# robot_pos_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.pose.position.x,
#                                                                                                        msg.pose.position.y))
# robot_vel_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.twist.linear.x,
#                                                                                                        msg.twist.linear.y))
# robot_acc_callback(msg, ts) = (push!(ts.t, Float64(msg.header.stamp)), push!(ts.xy, SVector{2,Float32}(msg.accel.linear.x,
#                                                                                                        msg.accel.linear.y))


function terminate_callback(msg::BoolMsg, planner_state::PlannerState)
    planner_state.terminate = msg.data
end

function reset_scenario_callback(msg::BoolMsg, planner_state::PlannerState)
    if msg.data
        planner_state.initialized = true
        planner_state.start_time = Float64(RobotOS.now())
        planner_state.computed_up_to_t = 0
        if RobotOS.get_param("robot_start_lane", "R") == "L"
            planner_state.robot_l0, planner_state.robot_lf = 1, 2
        else
            planner_state.robot_l0, planner_state.robot_lf = 2, 1
        end
        # if planner_state.human_buffer.pos.coefs[end][2] > -4    # HACK
        # planner_state.robot_l0, planner_state.robot_lf = 2, 1
        # else
        #     planner_state.robot_l0, planner_state.robot_lf = 2, 1
        # end
        # planner_state.robot_a0 = Tuple{Int,Int}[]
        # planner_state.robot_actions = [(1,planner_state.robot_l0) for i in 1:(start_wait + computation_interval)]
        rossleep(0.5)
        truncate!(planner_state.robot_des)
        truncate!(planner_state.human)
        truncate!(planner_state.robot)
        truncate!(planner_state.human_buffer)
        truncate!(planner_state.robot_buffer)
    end
end

init_node("traffic_weaving_planner", anonymous=true)
const terminate_sub = Subscriber{BoolMsg}("/terminate", terminate_callback, (planner_state,), queue_size=10)
const reset_sub = Subscriber{BoolMsg}("/reset", reset_scenario_callback, (planner_state,), queue_size=1)
# const reset_sim_sub = Subscriber{BoolMsg}("/reset_sim", reset_scenario_callback, (planner_state,), queue_size=10)

const human_pos_sub = Subscriber{PoseStamped}(human*"/pose", append_callback, (planner_state.human_buffer.pos,), queue_size=10)
const human_vel_sub = Subscriber{TwistStamped}(human*"/vel", append_callback, (planner_state.human_buffer.vel,), queue_size=10)
const human_acc_sub = Subscriber{AccelStamped}(human*"/acc", append_callback, (planner_state.human_buffer.acc,), queue_size=10)
const robot_pos_sub = Subscriber{PoseStamped}(robot*"/pose", append_callback, (planner_state.robot_buffer.pos,), queue_size=10)
const robot_vel_sub = Subscriber{TwistStamped}(robot*"/vel", append_callback, (planner_state.robot_buffer.vel,), queue_size=10)
const robot_acc_sub = Subscriber{AccelStamped}(robot*"/acc", append_callback, (planner_state.robot_buffer.acc,), queue_size=10)

const robot_traj_pub = Publisher{Float32MultiArrayStamped}("/robot/traj_plan", queue_size=10)
convert_to_3d_array(V::Vector{SVector{6,Float32}}) = reinterpret(Float32, copy(V), (6, length(V), 1))
convert_to_reversed_3d_array(V::Vector{SVector{6,Float32}}) = reshape(reinterpret(Float32, copy(V), (6, length(V)))', (1, length(V), 6))
function publish_robot_plan(robot_des::Vector{SVector{6,Float32}}, time_offset)
    msg = Float32MultiArrayStamped()
    msg.data = convert_to_3d_array(robot_des)
    msg.header.stamp = (time_offset isa Number ? RobotOS.Time(time_offset) : time_offset)
    publish(robot_traj_pub, msg)
end

include("utils/planner.jl")

function run(;start_wait=8, computation_interval=3)
    rate = RobotOS.Rate(100)

    time_scale_factor = RobotOS.get_param("time_scale_factor")
    dt = Float32(.1/time_scale_factor)

    reset_scenario_callback(BoolMsg(true), planner_state)
    planner_state.terminate = false
    while !planner_state.terminate
        @assert length(planner_state.human) == length(planner_state.robot)
        Tnew = min(last_common_time(planner_state.human_buffer), last_common_time(planner_state.robot_buffer))
        if isempty(planner_state.human)
            # t_now = Float64(RobotOS.now())
            # push!(planner_state.robot_buffer.pos, t_now, [-138.2, -6.0940447])
            # push!(planner_state.robot_buffer.pos, t_now, [28., 0.])
            # push!(planner_state.robot_buffer.pos, t_now, [0., 0.])
            Tfirst = max(first_common_time(planner_state.human_buffer), first_common_time(planner_state.robot_buffer), planner_state.start_time)
            println("still empty, $Tfirst, $Tnew")
            # println(planner_state.human_buffer.pos.knots)
            # println(planner_state.human_buffer.pos.coefs)
            # println(planner_state.robot_buffer)
            try
                if Tnew == -1 || Tfirst == -1 || Tfirst >= Tnew - 0.02 || planner_state.robot_buffer[Tnew-0.01][1] < -138
                    rossleep(0.3)
                    continue
                end
            catch
                println("brute-force catching our way past race conditions (1)")
                continue
            end
            A, B = Tfirst, Tnew
            Tstart = (A + B)/2
            for i in 1:10
                planner_state.robot_buffer[Tstart][1] < -138 ? (A = Tstart) : (B = Tstart)
                Tstart = (A + B)/2
            end
            println("ready to start")
            planner_state.start_time = Tstart
            initial_actions = [(1,planner_state.robot_l0) for i in 1:(start_wait + computation_interval)]
            planner_state.robot_des = propagate(xdd_choices, y_targets, planner_state.robot_buffer[planner_state.start_time], tibvp, initial_actions)
            publish_robot_plan(planner_state.robot_des, planner_state.start_time)
        end
        T = planner_state.start_time + (length(planner_state.human) - 1)*dt
        Tnew = min(last_common_time(planner_state.human_buffer), last_common_time(planner_state.robot_buffer))    # in case of race with /reset channel?

        race_condition_hit = false
        while T + dt < Tnew
            println("pushing")
            try
                push!(planner_state.human, planner_state.human_buffer[T + dt])
                push!(planner_state.robot, planner_state.robot_buffer[T + dt])
            catch
                println("brute-force catching our way past race conditions (2)")
                race_condition_hit = true
                break
            end
            if length(planner_state.robot) <= length(planner_state.robot_des)
                planner_state.robot_des[length(planner_state.robot)] = planner_state.robot[end]  # maybe unnecessary
            end
            T = T + dt
        end
        race_condition_hit && continue

        println("caught up")
        # t = floor(Int, (Float64(RobotOS.now()) - planner_state.start_time)/dt) + 1
        t = length(planner_state.human)
        t = t - (t % computation_interval)    # enforcing strict computation intervals
        if t <= 5
            rossleep(0.1)
            continue
        end

        println("sufficient history")

        # a0 = planner_state.robot_actions[t:t+computation_interval-1]
        if t+computation_interval > length(planner_state.robot_des) || t <= planner_state.computed_up_to_t
            rossleep(0.01)
            continue
        end
        a0_traj = planner_state.robot_des[t+1:t+computation_interval]
        a0 = ([a0_traj[i][5] for i in 1:computation_interval], [a0_traj[i][6] for i in 1:computation_interval])    # Ẍ and Ÿ; TODO: plumb targets instead of using previous Ÿ
        pfX, pfY = all_future_xy(xdd_choices, y_targets, planner_state.robot[t], tibvp, a0, 4, des_length=15)
        robot_in = convert_to_reversed_3d_array(planner_state.robot)
        human_in = convert_to_reversed_3d_array(planner_state.human)
        extras_in = zeros(Float32, 1, length(planner_state.robot), 0)
        predict!(robot_in, human_in, extras_in, [t], pfX[:,2:end,:], pfY[:,2:end,:], 64)
        for i in 1:15
            _t = t + i
            if _t <= length(planner_state.robot_des)
                planner_state.robot_des[_t] = SVector(pred_container.b[i,:]...)
            else
                push!(planner_state.robot_des, SVector(pred_container.b[i,:]...))
            end
        end
        publish_robot_plan(planner_state.robot_des, planner_state.start_time)
        planner_state.computed_up_to_t = t

        # TrafficWeavingPlanner.pred_container.y[4:6,1:2:end], TrafficWeavingPlanner.pred_container.y[4:6,2:2:end]
        # car1[1,t+1:t+15,:] = TrafficWeavingPlanner.pred_container.y
        # t = t + 3
        # car1_curr_state = car1[1,t,:]
        # a0 = (TrafficWeavingPlanner.pred_container.y[4:6,1:2:end], TrafficWeavingPlanner.pred_container.y[4:6,2:2:end])


        rossleep(rate)
    end
end

@spawn spin()

# mutable struct SimulatorState
#     reset::Bool
#     xh::SE2vState{Float64}
#     uh::AccelerationCurvatureControl{Float64}
# end

# function joystick_callback(msg::Joy, sim_state::SimulatorState)
#     if msg.buttons[1] == 1 && msg.buttons[2] == 1
#         sim_state.reset = true
#     end
#     sim_state.uh = AccelerationCurvatureControl(MAX_LONGITUDINAL_ACC*msg.axes[2],
#                                                 MAX_CURVATURE*msg.axes[1])
# end

# init_node("xbox_car", anonymous=true)

# const sim_state = SimulatorState(false, zeros(SE2vState{Float64}), zeros(AccelerationCurvatureControl{Float64}))
# const joystick_sub = Subscriber{Joy}("joy", joystick_callback, (sim_state,), queue_size=10)
# const human_vel_pub = Publisher{TwistStamped}("human/vel", queue_size=10)
# const human_accel_pub = Publisher{AccelStamped}("human/accel", queue_size=10)
# const tf_broadcaster = tf2_ros.TransformBroadcaster()

# function fill_2D_transform!(t::TransformStamped, x, y, θ, stamp = RobotOS.now())
#     t.header.stamp = stamp
#     t.transform.translation.x = x
#     t.transform.translation.y = y
#     t.transform.rotation.x,
#     t.transform.rotation.y,
#     t.transform.rotation.z,
#     t.transform.rotation.w = tf.transformations[:quaternion_from_euler](0.0, 0.0, θ)
#     t
# end

# function test()
#     rate = Rate(SIM_RATE)

#     world_header = Header()
#     world_header.frame_id = "world"
#     human_car_wrt_world = TransformStamped(world_header, "human_car", Transform())
#     vel_msg, accel_msg = TwistStamped(), AccelStamped()
#     vel_msg.header.frame_id = accel_msg.header.frame_id = "human_car"

#     sim_state.xh = zeros(SE2vState{Float64})
#     while !sim_state.reset
#         timestamp = RobotOS.now()
#         sim_state.xh = propagate(SimpleCarDynamics{1,0}(), sim_state.xh, StepControl(1/SIM_RATE, sim_state.uh))
#         fill_2D_transform!(human_car_wrt_world, sim_state.xh.x, sim_state.xh.y, sim_state.xh.θ, timestamp)

#         vel_msg.header.stamp = accel_msg.header.stamp = timestamp
#         vel_msg.twist.linear.x = sim_state.xh.v
#         accel_msg.accel.linear.x = sim_state.uh.a
#         accel_msg.accel.linear.y = sim_state.uh.κ*sim_state.xh.v^2

#         # p, o = pose_msg.pose.position, pose_msg.pose.orientation
#         # p.x, p.y = sim_state.xh.x, sim_state.xh.y
#         # o.x, o.y, o.z, o.w = tf.transformations[:quaternion_from_euler](0.0, 0.0, sim_state.xh.θ)
#         # v = vel_msg.twist.linear
#         # v.x, v.y = sim_state.xh.v*cos(sim_state.xh.θ), sim_state.xh.v*sin(sim_state.xh.θ)
#         # a = accel_msg.accel.linear
#         # a.x = sim_state.uh.a*cos(sim_state.xh.θ) - sim_state.uh.κ*sim_state.xh.v^2*sin(sim_state.xh.θ)
#         # a.y = sim_state.uh.a*sin(sim_state.xh.θ) + sim_state.uh.κ*sim_state.xh.v^2*cos(sim_state.xh.θ)

#         tf_broadcaster[:sendTransform](human_car_wrt_world)
#         # tf_broadcaster[:sendTransform]((sim_state.xh.x, sim_state.xh.y, 0.0),
#         #                                tf.transformations[:quaternion_from_euler](0.0, 0.0, sim_state.xh.θ),
#         #                                timestamp, "human_car", "world")
#         # publish(human_pose_pub, pose_msg)
#         publish(human_vel_pub, vel_msg)
#         publish(human_accel_pub, accel_msg)
#         rossleep(rate)
#     end
# end

# @spawn spin()

end # module
