#!/usr/bin/env python
from __future__ import division

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from osprey.msg import path
from safe_traffic_weaving.msg import VehicleTrajectory
from utils.markers import text_marker

class NominalTrajectoryVisualization:

    def __init__(self):
        rospy.init_node("nominal_trajectory_visualization", anonymous=True)
        self.pub = rospy.Publisher("/nominal_traj", MarkerArray, queue_size=10, latch=True)
        self.traj_marker = Marker()
        self.traj_marker.header.frame_id = "world"
        self.traj_marker.ns = "nominal_traj"
        self.traj_marker.type = Marker.LINE_STRIP
        self.traj_marker.scale.x = 0.5
        self.traj_marker.frame_locked = True
        self.lo_speed_marker = text_marker("world")
        self.hi_speed_marker = text_marker("world")
        self.hi_speed_marker.id = 1
        l = rospy.get_param("roadway/length")
        hwy_start_frac = rospy.get_param("hwy_start_frac")
        hwy_split_frac = rospy.get_param("hwy_split_frac")
        x_scale_factor = l*(hwy_split_frac - hwy_start_frac) / 135.0
        time_scale_factor = rospy.get_param("time_scale_factor", 0.5)
        self.min_speed = 18 * time_scale_factor * x_scale_factor
        self.max_speed = 40 * time_scale_factor * x_scale_factor

        self.markers = MarkerArray([self.traj_marker,
                                    self.lo_speed_marker,
                                    self.hi_speed_marker])
        rospy.Subscriber("/des_path", path, self.repub_path_callback)
        rospy.Subscriber("/des_traj", VehicleTrajectory, self.repub_traj_callback)
        rospy.spin()

    def repub_path_callback(self, msg):
        stamp = rospy.Time.now()
        for m in self.markers.markers:
            m.header.stamp = stamp
        lo_speed_idx = np.array(msg.Ux_des_mps).argmin()
        hi_speed_idx = np.array(msg.Ux_des_mps).argmax()
        self.lo_speed_marker.text = "%.2f" % msg.Ux_des_mps[lo_speed_idx]
        self.hi_speed_marker.text = "%.2f" % msg.Ux_des_mps[hi_speed_idx]
        self.lo_speed_marker.pose.position.x = msg.posE_m[lo_speed_idx]
        self.lo_speed_marker.pose.position.y = msg.posN_m[lo_speed_idx]
        self.hi_speed_marker.pose.position.x = msg.posE_m[hi_speed_idx]
        self.hi_speed_marker.pose.position.y = msg.posN_m[hi_speed_idx]

        self.traj_marker.points = []
        self.traj_marker.colors = []
        for x, y, v in zip(msg.posE_m, msg.posN_m, msg.Ux_des_mps):
            self.traj_marker.points.append(Point(x, y, 0.0))
            speed_frac = max(0.0, (v - self.min_speed) / (self.max_speed - self.min_speed))
            if speed_frac > 1:
                self.traj_marker.colors.append(ColorRGBA(1.0, 1.0, 0.0, 1.0))
            else:
                self.traj_marker.colors.append(ColorRGBA(1 - speed_frac, speed_frac, 0.0, 1.0))

        self.pub.publish(self.markers)

    def repub_traj_callback(self, msg):
        stamp = rospy.Time.now()
        for m in self.markers.markers:
            m.header.stamp = stamp
        lo_speed_idx = np.array(msg.V).argmin()
        hi_speed_idx = np.array(msg.V).argmax()
        self.lo_speed_marker.text = "%.2f" % msg.V[lo_speed_idx]
        self.hi_speed_marker.text = "%.2f" % msg.V[hi_speed_idx]
        self.lo_speed_marker.pose.position.x = msg.E[lo_speed_idx]
        self.lo_speed_marker.pose.position.y = msg.N[lo_speed_idx]
        self.hi_speed_marker.pose.position.x = msg.E[hi_speed_idx]
        self.hi_speed_marker.pose.position.y = msg.N[hi_speed_idx]

        self.traj_marker.points = []
        self.traj_marker.colors = []
        for x, y, v in zip(msg.E, msg.N, msg.V):
            self.traj_marker.points.append(Point(x, y, 0.0))
            speed_frac = max(0.0, (v - self.min_speed) / (self.max_speed - self.min_speed))
            if speed_frac > 1:
                self.traj_marker.colors.append(ColorRGBA(0.0, 1.0, 0.0, 1.0))
            else:
                self.traj_marker.colors.append(ColorRGBA(1 - speed_frac, speed_frac, 0.0, 1.0))

        self.pub.publish(self.markers)

if __name__ == '__main__':
    ntv = NominalTrajectoryVisualization()
