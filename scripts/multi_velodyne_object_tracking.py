#!/usr/bin/env python
import rospy
import numpy as np
import tf
import tf.msg
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Quaternion, AccelStamped, TwistStamped, TransformStamped, PointStamped, Point
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from auto_messages.msg import from_autobox
from std_msgs.msg import ColorRGBA

from utils.math_utils import *


class Position:
    def __init__(self, x=[0, 0, 0, 0, 0, 0]):
        self.x = x[0]
        self.y = x[1]
        self.xd = x[2]
        self.yd = x[3]
        self.xdd = x[4]
        self.ydd = x[5]

    def to_list(self):
        return [self.x, self.y, self.xd, self.yd, self.xdd, self.ydd]

class Orientation:
    def __init__(self, w=[0, 0, 0, 1], dtheta = 0, dtheta_prev = 0):
        self.w0 = w[0]
        self.w1 = w[1]
        self.w2 = w[2]
        self.w3 = w[3]
        self.dtheta = dtheta
        self.dtheta_prev = dtheta_prev

    def to_list(self):
        return [self.w0, self.w1, self.w2, self.w3]

class State:
    def __init__(self, pos = Position(), orien = Orientation(), frame=None):
        self.position = pos
        self.orientation = orien
        self.frame = frame

class x1lidar:

    def __init__(self, intensity_tol=100):
        self.intensity_tol = intensity_tol
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        rospy.init_node("visualize_lidar", anonymous=True)
        # rospy.Subscriber("from_veh", fromVehicle, self.egocar_callback)

        # in world frame
        self.trackedObject_pose_pub = rospy.Publisher("/tracked_object/pose", PoseStamped, queue_size=10)
        self.trackedObject_vel_pub = rospy.Publisher("/tracked_object/vel", TwistStamped, queue_size=10)
        self.trackedObject_accel_pub = rospy.Publisher("/tracked_object/acc", AccelStamped, queue_size=10)
        self.relevant_pc = rospy.Publisher("/tracked_object/pc_viz", Marker, queue_size=10)
        self.ellipse = rospy.Publisher("/tracked_object/ellipse_viz", Marker, queue_size=10)

        # in world frame
        # self.x1_marker_pub = rospy.Publisher("/x1/marker", Marker, queue_size=10)
        rospy.Subscriber("/x1/pose", PoseStamped, self.x1_position)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.initialize_trackedObject)
        
        # rospy.Subscriber("/M_HDL32/velodyne_points", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/M_intensity_filter/output", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)
        # rospy.Subscriber("/FL_VLP16/velodyne_points", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/FL_intensity_filter/output", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)
        # rospy.Subscriber("/FR_VLP16/velodyne_points", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/FR_intensity_filter/output", PointCloud2, self.pc2_callback, queue_size=1, buff_size=2**24)

        # pc array marker
        self.pc_marker = Marker()
        self.pc_marker.ns = "considered_pc"
        self.pc_marker.type = Marker.POINTS
        self.pc_marker.scale.x = 0.05
        self.pc_marker.scale.y = 0.05
        self.pc_marker.frame_locked = True

        # ellipse array marker
        self.ellipse_marker = Marker()
        self.ellipse_marker.header.frame_id = "/world"
        self.ellipse_marker.ns = "ellipse"
        self.ellipse_marker.type = Marker.LINE_STRIP
        self.ellipse_marker.scale.x = 0.01
        self.ellipse_marker.frame_locked = True
        self.ellipse_marker.color.g = 1
        self.ellipse_marker.color.a = 1

        # in world frame
        self.trackedObjectState = None        
        self.x1State = None

        self.initialize_flag = False
        self.x1_frame_init = False
        self.dt = None
        self.prev_time = 0      # to be overwritten
        self.curr_time = 0.1    # to be overwritten
        self.lost_count = 0
        self.processing = False

    def pose_twist_accelStamped_pub(self, state, pose_pub, vel_pub, accel_pub, header_frame_id, timestamp):
        self.tf_broadcaster.sendTransform((state.position.x, state.position.y, 0), 
                                           state.orientation.to_list(),
                                           timestamp,
                                           "tracked_object",
                                           header_frame_id)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = header_frame_id
        pose_msg.header.stamp = timestamp
        pose_msg.pose.position.x = state.position.x
        pose_msg.pose.position.y = state.position.y
        pose_msg.pose.orientation = Quaternion(*state.orientation.to_list())
        pose_pub.publish(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.frame_id = header_frame_id
        vel_msg.header.stamp = timestamp
        vel_msg.twist.linear.x = state.position.xd
        vel_msg.twist.linear.y = state.position.yd
        vel_msg.twist.angular.z = state.orientation.dtheta
        vel_pub.publish(vel_msg)

        accel_msg = AccelStamped()
        accel_msg.header.frame_id = header_frame_id
        accel_msg.header.stamp = timestamp
        accel_msg.accel.linear.x = state.position.xdd
        accel_msg.accel.linear.y = state.position.ydd
        accel_pub.publish(accel_msg)


    def x1_position(self, msg):
        # PoseStamped msg
        self.x1State = msg

    def initialize_trackedObject(self, msg):
        # transform nav_goal message to the world frame
        msg_time = msg.header.stamp
        msg_frame = msg.header.frame_id
        self.prev_time = msg.header.stamp.to_time()
        try:
            self.tf_listener.waitForTransform("/world", msg_frame, msg_time, rospy.Duration(.5))
            pose_world = self.tf_listener.transformPose("/world", msg)

        except:
            rospy.logwarn("Could not transform from vehicle base to World coordinates")

        # pose_world = self.tf_listener.transformPose('/world', msg)
        ori = Orientation([pose_world.pose.orientation.x, 
                           pose_world.pose.orientation.y, 
                           pose_world.pose.orientation.z, 
                           pose_world.pose.orientation.w])
        th = euler_from_quaternion(ori.to_list())
        pos = Position([pose_world.pose.position.x, pose_world.pose.position.y, np.cos(th[2]), np.sin(th[2]), 0, 0])


        self.trackedObjectState = State(pos, ori, '/world')
        # this publishes the transform, and this will enable the viz_vehicle file to plot the tracked_object
        self.pose_twist_accelStamped_pub(self.trackedObjectState, 
                                         self.trackedObject_pose_pub, 
                                         self.trackedObject_vel_pub,
                                         self.trackedObject_accel_pub,
                                         '/world', 
                                         msg_time)
        # flag to say that it has been initialized.
        self.initialize_flag = True
        

        # self.ellipse_marker.header.frame_id = msg_frame
        # self.ellipse_marker.points = []
        # center = np.array([[pos.x], [pos.y]])
        # ellipse_points = self.get_ellipse(3) + center
        # for i in range(ellipse_points.shape[-1]):
        #     self.ellipse_marker.points.append(Point(ellipse_points[0,i], ellipse_points[1,i], 0)) 
        # self.ellipse.publish(self.ellipse_marker)

        self.initialize_EKF()
        rospy.loginfo("Initialized the marker!")    
        initial_position = Position((self.x1State.pose.position.x, self.x1State.pose.position.y, 0, 0, 0, 0))
        initial_orientation = Orientation((self.x1State.pose.orientation.x, self.x1State.pose.orientation.y, self.x1State.pose.orientation.z, self.x1State.pose.orientation.w))
        self.initial_state = State(initial_position, initial_orientation, '/world')

        # make local frame. The frame where x1 is originally.
        self.tf_broadcaster.sendTransform((self.x1State.pose.position.x, self.x1State.pose.position.y, 0),
                                          initial_orientation.to_list(), 
                                          self.x1State.header.stamp,
                                          '/local',
                                          '/world')

    def get_ellipse(self, sd_tol, var):
        L = np.linalg.cholesky(var)
        LHinv = np.linalg.inv(L.T.conj())
        th = np.arange(0, 2*np.pi+1, 0.1)
        y1 = sd_tol * np.cos(th)
        y2 = sd_tol * np.sin(th)
        y = np.stack([y1,y2])
        return np.dot(LHinv, y)



    # tracking the other car
    def pc2_callback(self, msg, min_points = 2, vel_tol = 0.1, alpha = 0.95, sd = 3.0):
        intensity_max = 500
        
        if self.processing:
            rospy.logwarn("Callback is busy")
            return

        rospy.loginfo("Receiving points from velodyne %s", msg.header.frame_id)
        msg_time = msg.header.stamp
        msg_frame = msg.header.frame_id

        self.pc_marker.header.frame_id = msg_frame
        self.pc_marker.points = []
        self.pc_marker.colors = []
        self.ellipse_marker.points = []

        self.curr_time = msg_time.to_time()
        if self.dt:
            dt = self.curr_time - self.prev_time
        else: 
            dt = 0.1
        if dt < 0:
            rospy.logwarn("Message is too old")
            self.processing = False
            return
        else:
            self.dt = dt

        self.prev_time = msg_time.to_time()
        if self.initialize_flag:
            self.processing = True
            self.tf_broadcaster.sendTransform((self.initial_state.position.x, self.initial_state.position.y, 0.0),
                                               self.initial_state.orientation.to_list(), 
                                               msg_time,
                                               '/local',
                                               '/world')
            rospy.loginfo("Got velodyne points, getting x, y, z....")
            lidar_info = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "intensity"))
            x_list, y_list, intensities = [], [], []
            # in world frame
            xy_var = self.var[:2,:2]
            inv_xy_var = np.linalg.inv(xy_var)
            pt = PointStamped()
            pt.header.frame_id = '/world'
            pt.header.stamp = msg_time
            pt.point.x = self.trackedObjectState.position.x
            pt.point.y = self.trackedObjectState.position.y

            distance = np.hypot(self.trackedObjectState.position.x - self.x1State.pose.position.x, 
                                self.trackedObjectState.position.y - self.x1State.pose.position.y)
            sd_tol = 2*sd if distance > 5 else sd
            # trackedObjectPoints_velodyne = self.tf_listener.transformPoint("/velodyne", pt)
            try:
                self.tf_listener.waitForTransform(msg_frame, "/world", msg_time, rospy.Duration(.05))
                trackedObjectPoints_velodyne = self.tf_listener.transformPoint(msg_frame, pt)
                (trans, rot) = self.tf_listener.lookupTransform('/world', msg_frame, msg_time)
                th = euler_from_quaternion(rot)
                R = rot_mat(th[2])
                # variance for velodyne points in the velodyne frame but want to compute it in the world frame
                inv_var_rot = np.matmul(np.matmul(R, inv_xy_var),R.T)
            except:
                rospy.logwarn("Could not transform from World to Velodyne coordinates")
                self.processing = False
                return
   
            center = np.array([[self.trackedObjectState.position.x], [self.trackedObjectState.position.y]])
            ellipse_points = self.get_ellipse(sd_tol, inv_xy_var) + center
            for i in range(ellipse_points.shape[-1]):
                self.ellipse_marker.points.append(Point(ellipse_points[0,i], ellipse_points[1,i], 0)) 
            self.ellipse.publish(self.ellipse_marker)

            # trackedObjectPoints_velodyne = self.tf_listener.transformPoint(msg_frame, pt)

            for p in lidar_info:
                if p[2] > self.intensity_tol:
                    xy = np.array([[p[0] - trackedObjectPoints_velodyne.point.x], [p[1] - trackedObjectPoints_velodyne.point.y]])
                    #  Mahalanobis distance
                    dist = np.matmul(np.matmul(xy.T, inv_var_rot), xy)
                    # xy[0]**2*inv_var_rot[0,0] + xy[1]**2*inv_var_rot[1,1] + 2*xy[0]*xy[1]*inv_var_rot[1,0]
                    # rospy.logwarn("intensity tolerance met")
                    if dist < sd_tol**2: 
                        intensity_frac = max(p[2] / intensity_max, 1)
                        self.pc_marker.colors.append(ColorRGBA(1 - intensity_frac, intensity_frac, 0.0, 1.0))
                        self.pc_marker.points.append(Point(p[0], p[1], 0))
                        x_list.append(p[0])
                        y_list.append(p[1])
            self.relevant_pc.publish(self.pc_marker)


            if len(x_list) < min_points: 
                rospy.logwarn("%s did not receive points", msg.header.frame_id)
                self.lost_count += 1
                if self.lost_count > 3:
                    rospy.logwarn("Robot is lost in %s", msg.header.frame_id)
                # self.processing = False
                obs = None
            else: 
                self.lost_count = 0
                obs_velodyne = np.array([np.mean(x_list), np.mean(y_list)])
                obs_velodyne_point = PointStamped()
                obs_velodyne_point.header.frame_id = msg_frame
                obs_velodyne_point.header.stamp = msg_time
                obs_velodyne_point.point.x = obs_velodyne[0]
                obs_velodyne_point.point.y = obs_velodyne[1]
                # obs_world_point = self.tf_listener.transformPoint("/world", obs_velodyne_point)
                try:
                    self.tf_listener.waitForTransform("/world", msg_frame, msg_time, rospy.Duration(.05))
                    obs_world_point = self.tf_listener.transformPoint("/world", obs_velodyne_point)
                except:
                    rospy.logwarn("Could not transform from Velodyne to World coordinates")
                    self.processing = False
                    return

                # obs_world_point = self.tf_listener.transformPoint("/world", obs_velodyne_point)

                obs = np.array([obs_world_point.point.x, obs_world_point.point.y])

            mean, var = self.EKF(np.array(self.trackedObjectState.position.to_list()), self.var, obs)
            self.var = var
            pos = Position(list(mean))
            theta = np.arctan2(mean[3], mean[2])
            # theta0 = euler_from_quaternion(self.trackedObjectState.orientation.to_list())[2]
            # if angle changes too much, keep the same as before
            # if pos.xd**2 + pos.yd**2 < vel_tol and self.diff_angle(theta, theta0) > 0.1:
            # if self.diff_angle(theta, theta0) > 0.1:
            #     ori = Orientation(quaternion_from_euler(0.0, 0.0, theta0*alpha + theta*(1-alpha)))
            # else:
            ori = Orientation(quaternion_from_euler(0.0, 0.0, theta))

            # update the position and orientation of the tracked_object
            # self.trackedObjectState = State(pos, ori, '/world')
            self.trackedObjectState.position = pos
            self.trackedObjectState.orientation = ori
            self.pose_twist_accelStamped_pub(self.trackedObjectState, 
                                             self.trackedObject_pose_pub, 
                                             self.trackedObject_vel_pub, 
                                             self.trackedObject_accel_pub, 
                                             header_frame_id = '/world',
                                             timestamp=msg_time)
        self.processing = False

    def diff_angle(self, t1, t2):
        if np.sign(t1*t1) >= 0:
            return np.abs(t1 - t2)
        else:
            t = np.abs(t1 - t2)
            return min(t, np.pi*2 - t)

    def dAdu(self):
        dAdu = np.zeros([6, 2])
        dAdu[0,0] = 1.0/6.0 * self.dt**3
        dAdu[1,1] = 1.0/6.0 * self.dt**3
        dAdu[2,0] = 0.5 * self.dt**2
        dAdu[3,1] = 0.5 * self.dt**2
        dAdu[4,0] = self.dt
        dAdu[5,1] = self.dt
        return dAdu

    def dAdx(self):
        dAdx = np.diag(np.ones(6))
        dAdx[0,2] = self.dt
        dAdx[0,4] = 0.5*self.dt**2
        dAdx[1,3] = self.dt
        dAdx[1,5] = 0.5*self.dt**2
        dAdx[2,4] = self.dt
        dAdx[3,5] = self.dt
        return dAdx

    def obs_noise_covariance(self):
        return 0.09*np.eye(2)

    def process_noise_covariance(self):
        v = [0.000001, 0.000001, 0.000001, 0.000001, 0.25, 0.25]
        # v = np.ones(6)*0.01
        return np.diag(v)

    def step_dynamics(self, state, control= [0,0]):
        x, y, xd, yd, xdd, ydd = state[0], state[1], state[2], state[3], state[4], state[5]
        xddd, yddd = control[0], control[1]
        x += xd*self.dt + 0.5*xdd*self.dt**2 + 1.0/6.0*xddd*self.dt**3
        y += yd*self.dt + 0.5*ydd*self.dt**2 + 1.0/6.0*yddd*self.dt**3
        xd += xdd*self.dt + 0.5*xddd*self.dt**2
        yd += ydd*self.dt + 0.5*yddd*self.dt**2
        xdd += xddd*self.dt
        ydd += yddd*self.dt
        return np.array([x, y, xd, yd, xdd, ydd])

    def control_noise_in_control_space(self):
        return 64*np.eye(2)

    def observation_jacobian(self):
        j = np.zeros([2, 6])
        j[0,0] = 1.0
        j[1,1] = 1.0
        return j

    def observe(self, mean):
        return np.array([mean[0], mean[1]])

    def initialize_EKF(self):
        print("initializing EKF")
        self.var = np.diag([0.5, 0.5, 0.1, 0.1, 0.1, 0.1])
        self.R = self.process_noise_covariance()     # constant
        self.M = self.control_noise_in_control_space()   # constant
        self.H = self.observation_jacobian()
        self.O = self.obs_noise_covariance()

    def EKF(self, mean, var, obs, control = [0,0]):
        self.G = self.dAdx()
        self.V = self.dAdu()
        # mean is [6,1]
        # var is [6,6]
        # obs = [2,1]
        R, M, H, O = self.R, self.M, self.H, self.O
        # [6,6], [2,2], [2, 6], [2, 2]
        G, V = self.G, self.V
        # [6,6], [6,2]
        mean_bar = self.step_dynamics(mean, control)
        var_bar = G.dot(var).dot(G.T) + R*self.dt + V.dot(M).dot(V.T)*self.dt
        # [6,6][6,6][6,6] + [6,6] + [6,2][2,2][2,6]
        S = H.dot(var_bar).dot(H.T) + O
        # ([2,6][6,6][6,2] + [2,2])**-1 = [2,2]
        K = var_bar.dot(H.T).dot(np.linalg.inv(S))
        # [6,6][6,2][2,2]
        # K is [6,2]
        if obs is not None:
            mean_next = mean_bar + K.dot(obs - self.observe(mean_bar))
        else:
            mean_next = mean_bar
        # [6,1] + [6,2][2,1]
        # var_next = var_bar - np.matmul(K, np.matmul(H, var_bar))
        var_next = var_bar - K.dot(S).dot(K.T)
        # [6,6] - [6,2][2,6][6,6]
        return mean_next, var_next

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    args = rospy.myargv()
    intensity_tol = float(args[1]) if len(args) > 1 else 20
    lidar = x1lidar(intensity_tol=intensity_tol)
    lidar.run()


# just in case
# try:
#     self.tf_listener.waitForTransform("/world", msg_frame, msg_time, rospy.Duration(.5))
#     obs_world_point = self.tf_listener.transformPoint("/world", obs_velodyne_point)
# except:
#     rospy.logwarn("Could not transform from Velodyne to World coordinates")
