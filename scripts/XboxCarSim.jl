#!/usr/bin/env julia

module XboxCarSim

const MAX_LONGITUDINAL_ACC = 5.0
const MAX_LONGITUDINAL_BRAKE = -8.0
const MAX_CURVATURE = 1/20    # 1/(turning radius)
const SIM_RATE = 60

using PyCall
using RobotOS
using StaticArrays
using Rotations
using DifferentialDynamicsModels
using SimpleCarModels

@pyimport tf2_ros
@pyimport rospy
@rosimport safe_traffic_weaving.msg: XYThV
@rosimport sensor_msgs.msg: Joy
@rosimport std_msgs.msg: Bool, Header
@rosimport geometry_msgs.msg: Transform, TransformStamped, PoseStamped, TwistStamped, AccelStamped
@rosimport visualization_msgs.msg: Marker, MarkerArray
rostypegen()
import safe_traffic_weaving.msg: XYThV
import sensor_msgs.msg: Joy
import std_msgs.msg: BoolMsg, Header
import geometry_msgs.msg: Transform, TransformStamped, PoseStamped, TwistStamped, AccelStamped
import visualization_msgs.msg: Marker, MarkerArray

include("utils/utils.jl")

init_node("xbox_car", anonymous=true)

const terminate_pub = Publisher{BoolMsg}("/terminate", queue_size=10)
const reset_pub = Publisher{BoolMsg}("/reset", queue_size=1)
const reset_sim_pub = Publisher{BoolMsg}("/reset_sim", queue_size=1)
const xbox_car_pose_pub = Publisher{PoseStamped}("/xbox_car/pose", queue_size=10)    # w.r.t. /world frame
const xbox_car_vel_pub = Publisher{TwistStamped}("/xbox_car/vel", queue_size=10)     # w.r.t. /world frame
const xbox_car_acc_pub = Publisher{AccelStamped}("/xbox_car/acc", queue_size=10)     # w.r.t. /world frame
const xbox_car_xythv_pub = Publisher{XYThV}("/xbox_car/xythv", queue_size=10)
const xbox_car_marker_pub = Publisher{MarkerArray}("/xbox_car/viz", queue_size=10)
const xbox_car_init_marker_pub = Publisher{MarkerArray}("/xbox_car/init_viz", queue_size=10)
const tf_broadcaster = tf2_ros.TransformBroadcaster()
const tfBuffer = tf2_ros.Buffer()
const listener = tf2_ros.TransformListener(tfBuffer)

const time_scale_factor = RobotOS.get_param("time_scale_factor", 1.0)
const x_scale_factor = RobotOS.get_param("x_scale_factor", 1.0)

@show time_scale_factor
@show x_scale_factor

mutable struct SimulatorState
    terminate::Bool
    x::SE2vState{Float64}
    u::AccelerationCurvatureControl{Float64}
    x0::SE2vState{Float64}
    v0::Float64
    x0_offset_mode::Bool

    prev_joystick_time::Float64
end

function initial_state_callback(msg::XYThV, sim_state::SimulatorState)
    sim_state.x0 = SE2vState(XYThV.x, XYThV.y, XYThV.th, XYThV.v)
end

function x1_vel_callback(msg::TwistStamped, sim_state::SimulatorState)
    sim_state.v0 = norm([msg.twist.linear.x, msg.twist.linear.y])
end

function joystick_callback(msg::Joy, sim_state::SimulatorState)
    curr_joystick_time = Float64(RobotOS.get_rostime())
    dt = sim_state.prev_joystick_time == 0 ? 0.0 : curr_joystick_time - sim_state.prev_joystick_time
    sim_state.prev_joystick_time = curr_joystick_time

    # terminate (end joy stick node)
    if msg.buttons[1] == 1 && msg.buttons[2] == 1    # A and B (green and red)
        println("Terminate xbox controller")
        publish(terminate_pub, BoolMsg(true))
        sim_state.terminate = true
    end

    # toggle offset
    if msg.buttons[3] == 1                           # X (blue)
        println("Toggle offset mode")
        sim_state.x0_offset_mode = !sim_state.x0_offset_mode
    end


    if msg.buttons[4] == 1                           # Y (yellow)
        sim_state.x = sim_state.x0
        if sim_state.x0_offset_mode
            try
                x1_tf = tfBuffer[:lookup_transform]("world", "xbox_car_init", rospy.Time())
                x = x1_tf[:transform][:translation][:x]
                y = x1_tf[:transform][:translation][:y]
                θ = RotZYX(Quat(x1_tf[:transform][:rotation][:w],
                                x1_tf[:transform][:rotation][:x],
                                x1_tf[:transform][:rotation][:y],
                                x1_tf[:transform][:rotation][:z])).theta1
                v = sim_state.v0 + sim_state.x0.v
                sim_state.x = SE2vState(x, y, θ, v)
            end
        end
        sim_state.u = zeros(sim_state.u)
    end

    # robot right lane, human left lane
    if msg.buttons[5] == 1                           # left bumper
        println("Robot in right lane")
        RobotOS.set_param("robot_start_lane", "R")
        RobotOS.set_param("human_start_lane", "L")
    end
    # robot left lane, human right lane
    if msg.buttons[6] == 1                           # right bumper
        println("Robot in left lane")
        RobotOS.set_param("robot_start_lane", "L")
        RobotOS.set_param("human_start_lane", "R")
    end

    # reset sim
    if msg.buttons[7] == 1                           # select
        println("Reset experiment")
        for i in 1:5
            rate = Rate(100)
            publish(reset_sim_pub, BoolMsg(true))
            rossleep(rate)
        end
    end

    # reset path -- track straight line (at the start of the experiment)
    if msg.buttons[8] == 1                           # start
        println("Start experiment")
        publish(reset_pub, BoolMsg(true))
    end

    if sim_state.x0_offset_mode
        if norm(msg.axes[4:5]) > .1
            sim_state.x0 = SE2vState(sim_state.x0.x + msg.axes[5]*5*dt,
                                     sim_state.x0.y + msg.axes[4]*5*dt,
                                     sim_state.x0.θ,
                                     sim_state.x0.v)
        end
        if abs(msg.axes[8]) > .1
            sim_state.x0 = SE2vState(sim_state.x0.x,
                                     sim_state.x0.y,
                                     sim_state.x0.θ,
                                     sim_state.x0.v + .05*msg.axes[8])
        end
    end
    accel_frac = (1 - msg.axes[6])/2
    brake_frac = (1 - msg.axes[3])/2
    sim_state.u = AccelerationCurvatureControl((MAX_LONGITUDINAL_ACC*accel_frac + MAX_LONGITUDINAL_BRAKE*brake_frac)*x_scale_factor*time_scale_factor^2,
                                               MAX_CURVATURE*msg.axes[1]/x_scale_factor)
end



const sim_state = SimulatorState(false, zeros(SE2vState{Float64}), zeros(AccelerationCurvatureControl{Float64}), zeros(SE2vState{Float64}), 0.0, true, 0.0)
const x0_sub = Subscriber{XYThV}("/xbox_car/set_init", initial_state_callback, (sim_state,), queue_size=10)
const X1_sub = Subscriber{TwistStamped}("/x1/vel", x1_vel_callback, (sim_state,), queue_size=10)
const joystick_sub = Subscriber{Joy}("/joy", joystick_callback, (sim_state,), queue_size=10)

function fill_2D_transform!(t::TransformStamped, x, y, θ, stamp = RobotOS.now())
    t.header.stamp = stamp
    t.transform.translation.x = x
    t.transform.translation.y = y
    q = Quat(RotZYX(θ, 0, 0))    # tf.transformations[:quaternion_from_euler](0.0, 0.0, θ)
    t.transform.rotation.x = q.x
    t.transform.rotation.y = q.y
    t.transform.rotation.z = q.z
    t.transform.rotation.w = q.w
    t
end

function run(x0 = zeros(SE2vState{Float64}), marker_color="lime")
    rate = Rate(SIM_RATE)

    world_header = Header()
    world_header.frame_id = "world"
    xbox_car_wrt_world = TransformStamped(world_header, "xbox_car", Transform())
    init_wrt_xbox_car = TransformStamped(Header(), "xbox_car_init", Transform())

    pose_msg, vel_msg, acc_msg = PoseStamped(), TwistStamped(), AccelStamped()
    xythv_msg = XYThV()
    pose_msg.header.frame_id = vel_msg.header.frame_id = acc_msg.header.frame_id = "world"
    # pose_msg.pose.orientation.w = 1    # pose is at the origin in the /xbox_car frame
    xbox_car_marker = MarkerArray([car_marker("xbox_car", marker_color),
                                   text_marker("xbox_car", "white", 5.0)])
    init_marker = MarkerArray([car_marker("xbox_car_init", marker_color, .2),
                               text_marker("xbox_car_init", "white", 5.0)])

    sim_state.x = sim_state.x0 = x0
    sim_state.terminate = false
    while !sim_state.terminate
        timestamp = RobotOS.now()
        sim_state.x = propagate(SimpleCarDynamics{1,0}(), sim_state.x, StepControl(1/SIM_RATE, sim_state.u))
        fill_2D_transform!(xbox_car_wrt_world, sim_state.x.x, sim_state.x.y, sim_state.x.θ, timestamp)
        fill_2D_transform!(init_wrt_xbox_car, sim_state.x0.x, sim_state.x0.y, sim_state.x0.θ, timestamp)
        init_wrt_xbox_car.header.frame_id = sim_state.x0_offset_mode ? "x1" : "world"

        pose_msg.header.stamp = vel_msg.header.stamp = acc_msg.header.stamp = timestamp
        xbox_car_marker.markers[2].text = "$(@sprintf("%.2f", sim_state.x.v))"
        init_marker.markers[2].text = "$(@sprintf("%.2f", sim_state.v0))"
        if sim_state.x0_offset_mode
            init_marker.markers[2].text = init_marker.markers[2].text*(sim_state.x0.v >= 0 ? "+" : "-")*"$(@sprintf("%.2f", abs(sim_state.x0.v)))"
        end
        # init_marker.markers[2].text = "$(@sprintf("%.2f", sim_state.x0_offset_mode ? sim_state.v0 + sim_state.x0.v : sim_state.x0.v))"
        # init_marker.markers[2].text = "$(@sprintf("%.2f", sim_state.x0.v))"
        # vel_msg.twist.linear.x = sim_state.x.v
        # acc_msg.accel.linear.x = sim_state.u.a
        # acc_msg.accel.linear.y = sim_state.u.κ*sim_state.x.v^2

        p, o = pose_msg.pose.position, pose_msg.pose.orientation
        p.x, p.y = sim_state.x.x, sim_state.x.y
        q =  Quat(RotZYX(sim_state.x.θ, 0, 0))    # tf.transformations[:quaternion_from_euler](0.0, 0.0, sim_state.x.θ)
        o.x, o.y, o.z, o.w = q.x, q.y, q.z, q.w
        v = vel_msg.twist.linear
        v.x, v.y = sim_state.x.v*cos(sim_state.x.θ), sim_state.x.v*sin(sim_state.x.θ)
        a = acc_msg.accel.linear
        a.x = sim_state.u.a*cos(sim_state.x.θ) - sim_state.u.κ*sim_state.x.v^2*sin(sim_state.x.θ)
        a.y = sim_state.u.a*sin(sim_state.x.θ) + sim_state.u.κ*sim_state.x.v^2*cos(sim_state.x.θ)
        xythv_msg.x, xythv_msg.y, xythv_msg.th, xythv_msg.v =
            sim_state.x.x, sim_state.x.y, sim_state.x.θ, sim_state.x.v

        tf_broadcaster[:sendTransform](xbox_car_wrt_world)
        tf_broadcaster[:sendTransform](init_wrt_xbox_car)
        publish(xbox_car_pose_pub, pose_msg)
        publish(xbox_car_vel_pub, vel_msg)
        publish(xbox_car_acc_pub, acc_msg)
        publish(xbox_car_xythv_pub, xythv_msg)
        publish(xbox_car_marker_pub, xbox_car_marker)
        publish(xbox_car_init_marker_pub, init_marker)
        rossleep(rate)
    end
end

@spawn spin()

end # module

# if !isinteractive()
    using ArgParse
# end

if !isinteractive()
    using XboxCarSim

    function parse_commandline()
        s = ArgParseSettings()
        @add_arg_table s begin
            "--x0"
                arg_type = Float64
                default = 0.0
            "--y0"
                arg_type = Float64
                default = 0.0
            "--th0"
                arg_type = Float64
                default = 0.0
            "--v0"
                arg_type = Float64
                default = 0.0
            "--color"
                default = "lime"
        end
        parse_args(s)
    end
    parsed_args = parse_commandline()

    XboxCarSim.run(XboxCarSim.SE2vState(parsed_args["x0"],
                                        parsed_args["y0"],
                                        parsed_args["th0"],
                                        parsed_args["v0"]),
                   parsed_args["color"])
end
