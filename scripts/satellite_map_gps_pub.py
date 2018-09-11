#!/usr/bin/env python
from __future__ import division

import os
import rospy
import rospkg
import tf2_ros
from genpy.message import fill_message_args
from rostopic import file_yaml_arg
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import NavSatFix
from utils.gps import lla2enu, enu2lla

class SatMapGPS(object):
    def __init__(self, gps_ref, map_origin_frame):
        rospy.init_node("satellite_map_gps_publisher", anonymous=True)
        self.pub = rospy.Publisher("/sat_map_gps", NavSatFix, queue_size=10)
        self.msg = NavSatFix()
        fill_message_args(self.msg, next(file_yaml_arg(gps_ref)()))
        self.llaRef = (self.msg.latitude, self.msg.longitude, self.msg.altitude)
        self.map_origin_frame = map_origin_frame
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.tfBroadcaster = tf2_ros.StaticTransformBroadcaster()
        map_origin_tf = self.tfBuffer.lookup_transform("world", map_origin_frame, rospy.Time(0), rospy.Duration(5.0))
        map_origin_tf.transform.translation = Vector3()
        map_origin_tf.transform.rotation.x = -map_origin_tf.transform.rotation.x
        map_origin_tf.transform.rotation.y = -map_origin_tf.transform.rotation.y
        map_origin_tf.transform.rotation.z = -map_origin_tf.transform.rotation.z
        map_origin_tf.header.frame_id = map_origin_frame
        map_origin_tf.header.stamp = rospy.Time.now()
        map_origin_tf.child_frame_id = "sat_map_origin"
        self.tfBroadcaster.sendTransform(map_origin_tf)

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            try:
                map_origin_tf = self.tfBuffer.lookup_transform("world", self.map_origin_frame, rospy.Time(0), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue
            self.msg.header.stamp = map_origin_tf.header.stamp
            self.msg.latitude, self.msg.longitude, self.msg.altitude = enu2lla(self.llaRef, (map_origin_tf.transform.translation.x,
                                                                                             map_origin_tf.transform.translation.y,
                                                                                             map_origin_tf.transform.translation.z))
            self.pub.publish(self.msg)
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()
    gps_ref = os.path.join(os.path.dirname(__file__), "../rviz/{0}.gps".format(rospy.get_param(args[1])))
    map_origin_frame = args[2]

    print (gps_ref, map_origin_frame)
    s = SatMapGPS(gps_ref, map_origin_frame)
    s.run()
