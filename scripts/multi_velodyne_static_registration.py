#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import pcl # https://github.com/strawlab/python-pcl/
import sys
import yaml
import os
from sensor_msgs.msg import PointCloud2

def translation_as_list(t):
    return [t.x, t.y, t.z]

def quaternion_as_list(q):
    return [q.x, q.y, q.z, q.w]

def transform_and_convert_to_pcl(pc2_cloud, transform):
    trans = translation_as_list(transform.translation)
    quat = quaternion_as_list(transform.rotation)
    A = tf.transformations.translation_matrix(trans).dot(tf.transformations.quaternion_matrix(quat))
    raw_points_list = [A.dot((p[0], p[1], p[2], 1.0))[:3] for p in pc2.read_points(pc2_cloud, field_names=("x", "y", "z"), skip_nans=True)]
    points_list = [p for p in raw_points_list if (1 < p[2] < 4 and
                                                  2.5 < np.hypot(p[0], p[1]) < 20 and
                                                  np.abs(np.arctan2(p[1], p[0])) < np.pi/4)]

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(points_list)

    return pcl_data

def apply_delta_to_transform(D, transform):
    trans = translation_as_list(transform.translation)
    quat = quaternion_as_list(transform.rotation)
    A = tf.transformations.translation_matrix(trans).dot(tf.transformations.quaternion_matrix(quat))
    B = D.dot(A)
    return (tf.transformations.translation_from_matrix(B),
            tf.transformations.quaternion_from_matrix(B))

def plot_pcl(cloud, dims=(0,1), label=None):
    plt.scatter([p[dims[0]] for p in cloud.to_list()],
                [p[dims[1]] for p in cloud.to_list()],
                s=.1, label=label)

class MultiVelodyneRegistration:

    def __init__(self, output_filename):
        rospy.init_node("multi_velodyne_registration", anonymous=True)
        self.output_filename = output_filename
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.M_HDL32_PC2 = None
        self.FL_VLP16_PC2 = None
        self.FR_VLP16_PC2 = None
        self.M_HDL32_tf0 = None
        self.FL_VLP16_tf0 = None
        self.FR_VLP16_tf0 = None
        rospy.Subscriber("/M_HDL32/velodyne_points", PointCloud2, self.M_HDL32_cb, queue_size=1)
        rospy.Subscriber("/FL_VLP16/velodyne_points", PointCloud2, self.FL_VLP16_cb, queue_size=1)
        rospy.Subscriber("/FR_VLP16/velodyne_points", PointCloud2, self.FR_VLP16_cb, queue_size=1)

    def M_HDL32_cb(self, msg):
        if self.M_HDL32_PC2 is None:
            self.M_HDL32_PC2 = msg

    def FL_VLP16_cb(self, msg):
        if self.FL_VLP16_PC2 is None:
            self.FL_VLP16_PC2 = msg

    def FR_VLP16_cb(self, msg):
        if self.FR_VLP16_PC2 is None:
            self.FR_VLP16_PC2 = msg

    def run(self):
        rate = rospy.Rate(10)
        while (self.M_HDL32_PC2 is None or self.FL_VLP16_PC2 is None or self.FR_VLP16_PC2 is None or
               self.M_HDL32_tf0 is None or self.FL_VLP16_tf0 is None or self.FR_VLP16_tf0 is None):
            try:
                self.M_HDL32_tf0  = self.tfBuffer.lookup_transform("vehicle_base",
                                                                   "M_velodyne",
                                                                   rospy.Time(0))
                self.FL_VLP16_tf0 = self.tfBuffer.lookup_transform("vehicle_base",
                                                                   "FL_velodyne_rough",
                                                                   rospy.Time(0))
                self.FR_VLP16_tf0 = self.tfBuffer.lookup_transform("vehicle_base",
                                                                   "FR_velodyne_rough",
                                                                   rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass
            rate.sleep()
        M_HDL32_PCL  = transform_and_convert_to_pcl(self.M_HDL32_PC2,
                                                    self.M_HDL32_tf0.transform)
                                                    # [0.000, 0.000, 1.470],
                                                    # [0.000, 0.000, -0.707, 0.707])
        FL_VLP16_PCL = transform_and_convert_to_pcl(self.FL_VLP16_PC2,
                                                    self.FL_VLP16_tf0.transform)
                                                    # [2.130, 1.020, 0.660],
                                                    # [0.000, 0.000, 0.000, 1.000])
        FR_VLP16_PCL = transform_and_convert_to_pcl(self.FR_VLP16_PC2,
                                                    self.FR_VLP16_tf0.transform)
                                                    # [2.130, -1.020, 0.660],
                                                    # [0.000, 0.000, 0.000, 1.000])

        ICP = pcl.IterativeClosestPoint()
        FL_conv, FL_delta_transform, FL_aligned, FL_fitness = ICP.icp(FL_VLP16_PCL,
                                                                      M_HDL32_PCL)

        ICP = pcl.IterativeClosestPoint()
        FR_conv, FR_delta_transform, FR_aligned, FR_fitness = ICP.icp(FR_VLP16_PCL,
                                                                      M_HDL32_PCL)

        FL_VLP16_tf = apply_delta_to_transform(FL_delta_transform, self.FL_VLP16_tf0.transform)
        FR_VLP16_tf = apply_delta_to_transform(FR_delta_transform, self.FR_VLP16_tf0.transform)

        FL_VLP16_args = (" ".join(map(str, FL_VLP16_tf[0])) + " " +
                         " ".join(map(str, FL_VLP16_tf[1])) + " " +
                         "vehicle_base FL_velodyne 100")
        FR_VLP16_args = (" ".join(map(str, FR_VLP16_tf[0])) + " " +
                         " ".join(map(str, FR_VLP16_tf[1])) + " " +
                         "vehicle_base FR_velodyne 100")
        calibration_dict = dict(FL_VLP16_args=FL_VLP16_args, FR_VLP16_args=FR_VLP16_args)
        with open(self.output_filename, 'w') as f:
            yaml.dump(calibration_dict, f)

        plt.figure()
        plot_pcl(M_HDL32_PCL, label="M_HDL32")
        plot_pcl(FL_VLP16_PCL, label="FL_VLP16")
        plot_pcl(FR_VLP16_PCL, label="FR_VLP16")
        plt.title("uncalibrated")
        plt.axis("equal")
        leg = plt.legend()
        leg.legendHandles[0]._sizes = [30]
        leg.legendHandles[1]._sizes = [30]
        leg.legendHandles[2]._sizes = [30]
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.figure()
        plot_pcl(M_HDL32_PCL, label="M_HDL32")
        plot_pcl(FL_aligned, label="FL_VLP16")
        plot_pcl(FR_aligned, label="FR_VLP16")
        plt.title("calibrated")
        plt.axis("equal")
        leg = plt.legend()
        leg.legendHandles[0]._sizes = [30]
        leg.legendHandles[1]._sizes = [30]
        leg.legendHandles[2]._sizes = [30]
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.show()

if __name__ == '__main__':
    output_filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "multi_velodyne_calibration.yaml")
    mvr = MultiVelodyneRegistration(output_filename)
    mvr.run()
