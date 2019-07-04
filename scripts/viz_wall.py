#!/usr/bin/env python
from __future__ import division

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from utils.markers import colored_marker, straight_roadway_marker, line_segment_marker
from utils.math_utils import rot_unit_vectors, th_to_quat
from auto_messages.msg import from_autobox

class WallVisualization(object):
    def __init__(self, roadway_name):
        rospy.init_node("wall_visualization", anonymous=True)
        self.pub = rospy.Publisher("/{0}/wall".format(roadway_name), MarkerArray, queue_size=10, latch=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        roadway = rospy.get_param(roadway_name)
        wall_boundary_distance = rospy.get_param("wall_boundary_distance")
        x0 = np.array(roadway["start_mid"])
        w = roadway["lane_width"]
        l = roadway["length"]
        th = roadway["angle"]
        ns, nt = rot_unit_vectors(th)

        x0[0] += wall_boundary_distance * w * np.sin(th)
        x0[1] -= wall_boundary_distance * w * np.cos(th)
        wall = [-np.sin(th), np.cos(th), np.sin(th) * x0[0] - np.cos(th) * x0[1], th]

        self.wall_transform_msg = TransformStamped()
        self.wall_transform_msg.header.frame_id = "world"
        self.wall_transform_msg.child_frame_id = "right_wall"
        self.wall_transform_msg.transform.translation.x = x0[0]
        self.wall_transform_msg.transform.translation.y = x0[1]
        self.wall_transform_msg.transform.rotation = th_to_quat(roadway["angle"])


        right_wall_marker = colored_marker("lime", 1.0)
        right_wall_marker.header.frame_id = "right_wall"
        right_wall_marker.ns = "right_wall"
        right_wall_marker.type = Marker.LINE_STRIP
        right_wall_marker.scale.x = 0.5
        right_wall_marker.points = [Point(-500., 0., 0.), Point(500., 0., 0.)]
        right_wall_marker.frame_locked = True


        left_wall_marker = colored_marker("lime", 1.0)
        left_wall_marker.header.frame_id = "right_wall"
        left_wall_marker.ns = "left_wall"
        left_wall_marker.type = Marker.LINE_STRIP
        left_wall_marker.scale.x = 0.5
        dy = 2 * wall_boundary_distance * w
        left_wall_marker.points = [Point(-500., dy, 0.), Point(500., dy, 0.)]
        left_wall_marker.frame_locked = True

        self.marker_array = MarkerArray([right_wall_marker, left_wall_marker])


        rospy.Subscriber("/from_autobox", from_autobox, self.fromVehCallback, (wall))

    def fromVehCallback(self, msg, wall):
        # publish wall frame
        stamp = rospy.Time.now()
        a, b, c, th = wall
        y_rel = a * msg.E_m + b * msg.N_m + c
        x_wall = msg.E_m - a * y_rel
        y_wall = msg.N_m - b * y_rel
        self.wall_transform_msg.transform.translation.x = x_wall
        self.wall_transform_msg.transform.translation.y = y_wall
        self.wall_transform_msg.header.stamp = stamp
        self.tf_broadcaster.sendTransform(self.wall_transform_msg)
        for m in self.marker_array.markers:
            m.header.stamp = stamp
        self.pub.publish(self.marker_array)


    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()
    roadway_name = args[1] if len(args) > 1 else "roadway"
    rv = WallVisualization(roadway_name)
    rv.run()
