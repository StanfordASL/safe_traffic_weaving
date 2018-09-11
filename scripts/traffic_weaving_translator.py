#!/usr/bin/env python
import sys
import os
import h5py
import numpy as np
import rospy
import tf2_ros
from osprey.msg import path
from safe_traffic_weaving.msg import VehicleTrajectory, Float32MultiArrayStamped, PredictionOutput
from geometry_msgs.msg import TransformStamped, PoseStamped, TwistStamped, AccelStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray, ColorRGBA
from utils.numpy_ros import numpy_to_multiarray, multiarray_to_numpy
from utils.math_utils import rot_mat, rot_unit_vectors, rotate_v

def scale_time(A, s):
    return A*np.reshape([1, 1, s, s, s**2, s**2], (6,1,1))

def scale_x(A, s):
    return A*np.reshape([s, 1, s, 1, s, 1], (6,1,1))

def scale_y(A, s):
    return A*np.reshape([1, s, 1, s, 1, s], (6,1,1))

def set_x0_y0_th0(A, x0=0, y0=0, th0=0):
    T = A.shape[1]
    A = A.copy()
    A[0,:,:] = A[0,:,:] - A[0,0,:]
    A[1,:,:] = A[1,:,:] - A[1,0,:]
    A = np.reshape(np.matmul(rot_mat(th0), np.reshape(A, (3,2,-1))), (6,T,-1))
    A[0,:,:] = A[0,:,:] + x0
    A[1,:,:] = A[1,:,:] + y0
    return A

def car_array_to_path(A):
    N = A.shape[1]
    x = A[0,:,0]
    y = A[1,:,0]
    xd = A[2,:,0]
    yd = A[3,:,0]
    xdd = A[4,:,0]
    ydd = A[5,:,0]
    th = np.arctan2(yd, xd)
    s = np.insert(np.cumsum(np.hypot(np.diff(x), np.diff(y))), 0, 0)
    v = np.hypot(xd, yd)
    a = (xd*xdd + yd*ydd)/v
    k = (xd*ydd - xdd*yd)/(v**3)

    path_msg = path()
    path_msg.s_m = s
    path_msg.posE_m = x
    path_msg.posN_m = y
    path_msg.Psi_rad = th - np.pi/2
    path_msg.k_1pm = k
    path_msg.grade_rad = np.zeros(N)
    path_msg.edge_L_m = 4*np.ones(N)
    path_msg.edge_R_m = -4*np.ones(N)
    path_msg.Ux_des_mps = v
    path_msg.Ax_des_mps2 = a
    path_msg.isOpen = 1
    return path_msg

def car_array_to_VehicleTrajectory(A, time_scale_factor):
    N = A.shape[1]
    x = A[0,:,0]
    y = A[1,:,0]
    xd = A[2,:,0]
    yd = A[3,:,0]
    xdd = A[4,:,0]
    ydd = A[5,:,0]
    th = np.arctan2(yd, xd)
    s = np.insert(np.cumsum(np.hypot(np.diff(x), np.diff(y))), 0, 0)
    v = np.hypot(xd, yd)
    a = (xd*xdd + yd*ydd)/v
    k = (xd*ydd - xdd*yd)/(v**3)

    traj_msg = VehicleTrajectory()
    traj_msg.t = np.arange(N)*0.1/time_scale_factor
    traj_msg.s = s
    traj_msg.V = v
    traj_msg.A = a
    traj_msg.E = x
    traj_msg.N = y
    traj_msg.heading = th - np.pi/2
    traj_msg.curvature = k
    traj_msg.grade = np.zeros(N)
    traj_msg.bank = np.zeros(N)
    traj_msg.edge_L = 4*np.ones(N)
    traj_msg.edge_R = -4*np.ones(N)
    return traj_msg

class TrafficWeavingTranslator:

    def __init__(self):
        rospy.init_node("traffic_weaving_translator", anonymous=True)
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        roadway = rospy.get_param("roadway")
        x0 = np.array(roadway["start_mid"])
        w = roadway["lane_width"]
        l = roadway["length"]
        th = roadway["angle"]
        ns, nt = rot_unit_vectors(th)

        self.time_scale_factor = rospy.get_param("time_scale_factor", 0.5)
        self.hwy_start_frac = rospy.get_param("hwy_start_frac")
        self.hwy_split_frac = rospy.get_param("hwy_split_frac")
        self.x_scale_factor = l*(self.hwy_split_frac - self.hwy_start_frac) / 135.0
        rospy.set_param("x_scale_factor", self.x_scale_factor)
        self.y_scale_factor = w / 4.0
        self.origin = x0 + self.hwy_split_frac*l*ns + w*nt
        self.x_unit_vector = ns
        self.y_unit_vector = nt
        self.lane_th = th

        self.HUMAN = rospy.get_param("human", "/xbox_car")
        self.ROBOT = rospy.get_param("robot", "/x1")
        self.planner_human_pos_pub = rospy.Publisher("/human/pose", PoseStamped, queue_size=10)
        self.planner_human_vel_pub = rospy.Publisher("/human/vel", TwistStamped, queue_size=10)
        self.planner_human_acc_pub = rospy.Publisher("/human/acc", AccelStamped, queue_size=10)
        self.planner_robot_pos_pub = rospy.Publisher("/robot/pose", PoseStamped, queue_size=10)
        self.planner_robot_vel_pub = rospy.Publisher("/robot/vel", TwistStamped, queue_size=10)
        self.planner_robot_acc_pub = rospy.Publisher("/robot/acc", AccelStamped, queue_size=10)
        self.human_pos_sub = rospy.Subscriber(self.HUMAN + "/pose", PoseStamped, self.human_pos_cb)
        self.human_vel_sub = rospy.Subscriber(self.HUMAN + "/vel", TwistStamped, self.human_vel_cb)
        self.human_acc_sub = rospy.Subscriber(self.HUMAN + "/acc", AccelStamped, self.human_acc_cb)
        self.robot_pos_sub = rospy.Subscriber(self.ROBOT + "/pose", PoseStamped, self.robot_pos_cb)
        self.robot_vel_sub = rospy.Subscriber(self.ROBOT + "/vel", TwistStamped, self.robot_vel_cb)
        self.robot_acc_sub = rospy.Subscriber(self.ROBOT + "/acc", AccelStamped, self.robot_acc_cb)

        self.robot_path_pub = rospy.Publisher("/des_path", path, queue_size=10)
        self.robot_traj_pub = rospy.Publisher("/des_traj", VehicleTrajectory, queue_size=10)
        self.robot_path_plan_sub = rospy.Subscriber("/robot/path_plan", Float32MultiArray, self.translate_plan)
        self.robot_traj_plan_sub = rospy.Subscriber("/robot/traj_plan", Float32MultiArrayStamped, self.translate_traj)

        self.preds_marker = Marker()
        self.preds_marker.header.frame_id = "world"
        self.preds_marker.ns = "human_predictions"
        self.preds_marker.type = Marker.LINE_LIST
        self.preds_marker.scale.x = 0.2
        self.preds_marker.frame_locked = True
        self.min_speed = 18 * self.time_scale_factor * self.x_scale_factor
        self.max_speed = 40 * self.time_scale_factor * self.x_scale_factor
        self.viz_human_predictions_pub = rospy.Publisher("/human_predictions", Marker, queue_size=10, latch=True)
        self.prediction_output_sub = rospy.Subscriber("/prediction_output", PredictionOutput, self.viz_human_predictions)

        print "Time scale factor: ", self.time_scale_factor
        print "x scale factor: ", self.x_scale_factor
        print "y scale factor: ", self.y_scale_factor

    def human_pos_cb(self, msg):
        self.planner_human_pos_pub.publish(self.world_to_planner_pos(msg))

    def human_vel_cb(self, msg):
        self.planner_human_vel_pub.publish(self.world_to_planner_vel(msg))

    def human_acc_cb(self, msg):
        self.planner_human_acc_pub.publish(self.world_to_planner_acc(msg))

    def robot_pos_cb(self, msg):
        self.planner_robot_pos_pub.publish(self.world_to_planner_pos(msg))

    def robot_vel_cb(self, msg):
        self.planner_robot_vel_pub.publish(self.world_to_planner_vel(msg))

    def robot_acc_cb(self, msg):
        self.planner_robot_acc_pub.publish(self.world_to_planner_acc(msg))

    def world_to_planner_pos(self, msg):
        xy_world = np.array([msg.pose.position.x, msg.pose.position.y])
        xy_lane = xy_world - self.origin
        xu, yu = rotate_v(xy_lane, -self.lane_th)
        msg.pose.position.x = xu / self.x_scale_factor
        msg.pose.position.y = yu / self.y_scale_factor
        return msg

    def world_to_planner_vel(self, msg):
        xy_world = np.array([msg.twist.linear.x, msg.twist.linear.y])
        xu, yu = rotate_v(xy_world, -self.lane_th)
        msg.twist.linear.x = (xu / self.x_scale_factor) / self.time_scale_factor
        msg.twist.linear.y = (yu / self.y_scale_factor) / self.time_scale_factor
        return msg

    def world_to_planner_acc(self, msg):
        xy_world = np.array([msg.accel.linear.x, msg.accel.linear.y])
        xu, yu = rotate_v(xy_world, -self.lane_th)
        msg.accel.linear.x = (xu / self.x_scale_factor) / self.time_scale_factor**2
        msg.accel.linear.y = (yu / self.y_scale_factor) / self.time_scale_factor**2
        return msg

    def translate_plan(self, msg):
        A = multiarray_to_numpy(msg)
        if not A.size:    # guard against empty /robot/plan
            return
        A_shape_scaled = scale_y(scale_x(A, self.x_scale_factor), self.y_scale_factor)
        A_time_scaled = scale_time(A_shape_scaled, self.time_scale_factor)
        xy0 = self.origin + A_time_scaled[0,0,0]*self.x_unit_vector + A_time_scaled[1,0,0]*self.y_unit_vector
        th0 = np.arctan2(A_time_scaled[3,0,0], A_time_scaled[2,0,0]) + self.lane_th
        A = set_x0_y0_th0(A_time_scaled, xy0[0], xy0[1], th0)

        path_msg = car_array_to_path(A)
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = rospy.get_rostime()
        rospy.loginfo("last velocity profile %.1f", path_msg.Ux_des_mps[-1])
        self.robot_path_pub.publish(path_msg)

    def translate_traj(self, msg):
        A = multiarray_to_numpy(msg.data)
        if not A.size:    # guard against empty /robot/plan
            return
        A_shape_scaled = scale_y(scale_x(A, self.x_scale_factor), self.y_scale_factor)
        A_time_scaled = scale_time(A_shape_scaled, self.time_scale_factor)
        xy0 = self.origin + A_time_scaled[0,0,0]*self.x_unit_vector + A_time_scaled[1,0,0]*self.y_unit_vector
        th0 = np.arctan2(A_time_scaled[3,0,0], A_time_scaled[2,0,0]) + self.lane_th
        A = set_x0_y0_th0(A_time_scaled, xy0[0], xy0[1], th0)

        traj_msg = car_array_to_VehicleTrajectory(A, self.time_scale_factor)
        traj_msg.header = msg.header
        rospy.loginfo("last velocity profile %.1f", traj_msg.V[-1])
        self.robot_traj_pub.publish(traj_msg)

    def viz_human_predictions(self, msg):
        self.preds_marker.points = []
        self.preds_marker.colors = []
        Y = multiarray_to_numpy(msg.y)
        Z = multiarray_to_numpy(msg.z)
        Y = np.transpose(Y, (2,1,0))
        Y_shape_scaled = scale_y(scale_x(Y, self.x_scale_factor), self.y_scale_factor)
        Y_time_scaled = scale_time(Y_shape_scaled, self.time_scale_factor)
        xy0 = self.origin + Y_time_scaled[0,0,0]*self.x_unit_vector + Y_time_scaled[1,0,0]*self.y_unit_vector
        th0 = self.lane_th
        Y = set_x0_y0_th0(Y_time_scaled, xy0[0], xy0[1], th0)
        Y = np.transpose(Y, (2,1,0))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]-1):
                self.preds_marker.points.append(Point(Y[i,j,  0], Y[i,j,  1], 0.0))
                self.preds_marker.points.append(Point(Y[i,j+1,0], Y[i,j+1,1], 0.0))
                v1 = np.hypot(Y[i,j,2], Y[i,j,3])
                v2 = np.hypot(Y[i,j+1,2], Y[i,j+1,3])
                speed_frac1 = max(0.0, (v1 - self.min_speed) / (self.max_speed - self.min_speed))
                speed_frac2 = max(0.0, (v2 - self.min_speed) / (self.max_speed - self.min_speed))
                if speed_frac1 > 1:
                    self.preds_marker.colors.append(ColorRGBA(0.0, 1.0, 0.0, 1.0))
                else:
                    self.preds_marker.colors.append(ColorRGBA(1 - speed_frac1, speed_frac1, 0.0, 1.0))
                if speed_frac2 > 1:
                    self.preds_marker.colors.append(ColorRGBA(0.0, 1.0, 0.0, 1.0))
                else:
                    self.preds_marker.colors.append(ColorRGBA(1 - speed_frac2, speed_frac2, 0.0, 1.0))
        self.viz_human_predictions_pub.publish(self.preds_marker)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    twt = TrafficWeavingTranslator()
    twt.run()
