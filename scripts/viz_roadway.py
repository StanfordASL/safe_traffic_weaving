#!/usr/bin/env python
from __future__ import division

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from utils.markers import straight_roadway_marker, line_segment_marker
from utils.math_utils import rot_unit_vectors, th_to_quat

class RoadwayVisualization(object):
    def __init__(self, roadway_name, hwy_start_frac, hwy_split_frac):
        rospy.init_node("roadway_visualization", anonymous=True)
        self.pub = rospy.Publisher("/{0}/visualization".format(roadway_name), MarkerArray, queue_size=10, latch=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        roadway = rospy.get_param(roadway_name)
        x0 = np.array(roadway["start_mid"])
        w = roadway["lane_width"]
        l = roadway["length"]
        ns, nt = rot_unit_vectors(roadway["angle"])

        road_marker = straight_roadway_marker(roadway)
        start_marker = line_segment_marker(x0 + hwy_start_frac*l*ns - 10*nt, x0 + hwy_start_frac*l*ns + 10*nt, "start", "lime")
        split_marker = line_segment_marker(x0 + hwy_split_frac*l*ns - 10*nt, x0 + hwy_split_frac*l*ns + 10*nt, "end", "red")
        self.markers = MarkerArray([road_marker, start_marker, split_marker])

        self.transform_msg = TransformStamped()
        self.transform_msg.header.frame_id = "world"
        self.transform_msg.child_frame_id = "/{0}/start_mid".format(roadway_name)
        self.transform_msg.transform.translation.x = x0[0]
        self.transform_msg.transform.translation.y = x0[1]
        self.transform_msg.transform.rotation = th_to_quat(roadway["angle"])

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            for m in self.markers.markers:
                m.header.stamp = stamp
            self.transform_msg.header.stamp = stamp
            self.pub.publish(self.markers)
            self.tf_broadcaster.sendTransform(self.transform_msg)
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()
    roadway_name = args[1] if len(args) > 1 else "roadway"
    hwy_start_frac = float(args[2]) if len(args) > 2 else rospy.get_param("hwy_start_frac", .2)
    hwy_split_frac = float(args[3]) if len(args) > 3 else rospy.get_param("hwy_split_frac", .8)

    rv = RoadwayVisualization(roadway_name, hwy_start_frac, hwy_split_frac)
    rv.run()
