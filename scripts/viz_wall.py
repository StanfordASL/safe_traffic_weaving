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
        self.pub = rospy.Publisher("/{0}/wall".format(roadway_name), Marker, queue_size=10, latch=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        roadway = rospy.get_param(roadway_name)
        x0 = np.array(roadway["start_mid"])
        w = roadway["lane_width"]
        l = roadway["length"]
        th = roadway["angle"]
        ns, nt = rot_unit_vectors(th)

        road_marker = straight_roadway_marker(roadway)

        x0[0] += 1.2 * w * np.sin(th)
        x0[1] -= 1.2 * w * np.cos(th)
        x1 = x0 + ns * w
        wall = [-np.sin(th), np.cos(th), np.sin(th) * x0[0] - np.cos(th) * x0[1], th]

        self.wall_transform_msg = TransformStamped()
        self.wall_transform_msg.header.frame_id = "world"
        self.wall_transform_msg.child_frame_id = "wall"
        self.wall_transform_msg.transform.translation.x = x0[0]
        self.wall_transform_msg.transform.translation.y = x0[1]
        self.wall_transform_msg.transform.rotation = th_to_quat(roadway["angle"])

        self.marker = colored_marker("lime", 1.0)
        self.marker.header.frame_id = "wall"
        self.marker.ns = "wall"
        self.marker.type = Marker.LINE_STRIP
        self.marker.scale.x = 0.5
        self.marker.points = [Point(-5., 0., 0.), Point(5., 0., 0.)]
        self.marker.frame_locked = True

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

        self.marker.header.stamp = stamp
        self.pub.publish(self.marker)


    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()
    roadway_name = args[1] if len(args) > 1 else "roadway"
    rv = WallVisualization(roadway_name)
    rv.run()
