#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped, TwistStamped, AccelStamped
from auto_messages.msg import from_autobox
from utils.math_utils import rotate, th_to_quat, TranslationalState, AngularState

rospy.init_node("x1_state_publisher", anonymous=True)
tf_broadcaster = tf2_ros.TransformBroadcaster()
x1_pose_pub    = rospy.Publisher("/x1/pose", PoseStamped, queue_size=10)
x1_vel_pub     = rospy.Publisher("/x1/vel", TwistStamped, queue_size=10)
x1_accel_pub   = rospy.Publisher("/x1/acc", AccelStamped, queue_size=10)

def fromVehCallback(msg):
    timestamp = rospy.get_rostime()
    x = msg.E_m
    y = msg.N_m
    th = msg.psi_rad + np.pi/2.0
    xd, yd = rotate(msg.ux_mps, msg.uy_mps, th)
    xdd, ydd = rotate(msg.ax_mps2, msg.ay_mps2, th)

    x1_ts = TranslationalState(x, y, xd, yd, xdd, ydd)    # x1 in world frame
    x1_as = AngularState(th, msg.r_radps)

    publish_Transform_Pose_Twist_AccelStamped(x1_ts, x1_as, timestamp)

def publish_Transform_Pose_Twist_AccelStamped(trans, ang, timestamp, header_frame_id='world'):
    transform_msg = TransformStamped()
    transform_msg.header.frame_id = header_frame_id
    transform_msg.header.stamp = timestamp
    transform_msg.child_frame_id = "x1"
    transform_msg.transform.translation.x = trans.x
    transform_msg.transform.translation.y = trans.y
    transform_msg.transform.rotation = th_to_quat(ang.th)
    tf_broadcaster.sendTransform(transform_msg)

    pose_msg = PoseStamped()
    pose_msg.header = transform_msg.header
    pose_msg.pose.position.x = trans.x
    pose_msg.pose.position.y = trans.y
    pose_msg.pose.orientation = th_to_quat(ang.th)
    x1_pose_pub.publish(pose_msg)

    vel_msg = TwistStamped()
    vel_msg.header = transform_msg.header
    vel_msg.twist.linear.x = trans.xd
    vel_msg.twist.linear.y = trans.yd
    vel_msg.twist.angular.z = ang.thd
    x1_vel_pub.publish(vel_msg)

    accel_msg = AccelStamped()
    accel_msg.header = transform_msg.header
    accel_msg.accel.linear.x = trans.xdd
    accel_msg.accel.linear.y = trans.ydd
    x1_accel_pub.publish(accel_msg)

rospy.Subscriber("/from_autobox", from_autobox, fromVehCallback)

if __name__ == '__main__':
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
