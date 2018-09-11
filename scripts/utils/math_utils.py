from __future__ import division

import numpy as np
from collections import namedtuple
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion

def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def rot_mat(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th), np.cos(th)]])

def rot_unit_vectors(th):
    R = rot_mat(th)
    return R[:,0], R[:,1]

def rotate(x, y, th):
    return np.array([x*np.cos(th) - y*np.sin(th),
                     x*np.sin(th) + y*np.cos(th)])

def rotate_v(x, th):
    return np.dot(rot_mat(th), x)

def th_to_quat(th):
    return Quaternion(*quaternion_from_euler(0.0, 0.0, th))

def quat_to_th(quat):
    return euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))[2]

TranslationalState = namedtuple("TranslationalState", ["x", "y", "xd", "yd", "xdd", "ydd"])
AngularState = namedtuple("AngularState", ["th", "thd"])
