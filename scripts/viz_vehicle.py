#!/usr/bin/env python
from __future__ import division

import rospy
from visualization_msgs.msg import Marker
from utils.math_utils import int_or_float
from utils.markers import car_marker

VTD_CAR_X = 4.22100019455   # parameters corresponding to VTD simulated car
VTD_CAR_Y = 1.76199996471
VTD_CAR_dX = 1.3654999733

class VehicleVisualization(object):
    def __init__(self, name, X, Y, dX, color):
        rospy.init_node("vehicle_visualization", anonymous=True)
        self.pub = rospy.Publisher("/{0}/visualization".format(name), Marker, queue_size=10)
        self.marker = car_marker(name, color=color, CAR_X=X, CAR_Y=Y, CAR_dX=dX)

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.marker.header.stamp = rospy.Time.now()
            self.pub.publish(self.marker)
            rate.sleep()


if __name__ == "__main__":
    args = rospy.myargv()
    name = args[1]
    X = float(args[2]) if len(args) > 2 else VTD_CAR_X
    Y = float(args[3]) if len(args) > 3 else VTD_CAR_Y
    dX = float(args[4]) if len(args) > 4 else VTD_CAR_dX
    color = args[5] if len(args) > 5 else "red"
    color = (int_or_float(args[5]),
             int_or_float(args[6]),
             int_or_float(args[7])) if len(args) > 7 else color

    vv = VehicleVisualization(name, X, Y, dX, color)
    vv.run()
