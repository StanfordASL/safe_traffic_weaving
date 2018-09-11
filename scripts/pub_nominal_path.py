#!/usr/bin/env python
import rospy
import h5py
import numpy as np
from utils.numpy_ros import numpy_to_multiarray, multiarray_to_numpy
from std_msgs.msg import Float32MultiArray, Bool

R_path = np.array([[-200., -140.,   0.],    # x
                   [ -6.5,  -6.5, -6.5],    # y
                   [  12.,   28.,  28.],    # xd
                   [   0.,    0.,   0.],    # yd
                   [16./3,    0.,   0.],    # xdd
                   [   0.,    0.,   0.]]).reshape((6,-1,1))

L_path = np.array([[-200., -140.,   0.],    # x
                   [ -1.5,  -1.5, -1.5],    # y
                   [  12.,   28.,  28.],    # xd
                   [   0.,    0.,   0.],    # yd
                   [16./3,    0.,   0.],    # xdd
                   [   0.,    0.,   0.]]).reshape((6,-1,1))

class StraightPath:

    def __init__(self):
        rospy.init_node("straight_path", anonymous=True)
        self.path_pub = rospy.Publisher("/robot/path_plan", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/reset", Bool, self.pub_straight_path)

    def pub_straight_path(self, msg):
        if msg.data:
            start_lane = rospy.get_param("robot_start_lane", "R")
            A_path = R_path if start_lane == "R" else L_path
            for i in range(5):
                self.path_pub.publish(numpy_to_multiarray(A_path.astype(np.float32)))
                rospy.sleep(.1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    sp = StraightPath()
    sp.run()
