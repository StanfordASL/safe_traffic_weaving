from __future__ import division

import rospy
import numpy as np
from matplotlib.colors import get_named_colors_mapping, to_rgb
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from math_utils import rot_unit_vectors

COLOR_MAP = get_named_colors_mapping()

def color_to_rgb_float(color):
    if isinstance(color, str):
        color = to_rgb(COLOR_MAP[color])
    if isinstance(color, tuple) and isinstance(color[0], int):
        color = (color[0]/255, color[1]/255, color[2]/255)
    return color

def colored_marker(color, alpha=1.0):
    color = color_to_rgb_float(color)
    m = Marker()
    m.color.r = color[0]
    m.color.g = color[1]
    m.color.b = color[2]
    m.color.a = alpha
    m.pose.orientation.w = 1.0
    return m

def car_marker(body_frame_id, color = "red", alpha=1.0, CAR_X = 4.22100019455, CAR_Y = 1.76199996471, CAR_dX = 1.3654999733):
    m = colored_marker(color, alpha)
    m.header.frame_id = body_frame_id
    m.ns = "bounding_box"
    m.id = 0
    m.type = Marker.CUBE
    m.pose.position.x = CAR_dX
    m.scale.x = CAR_X
    m.scale.y = CAR_Y
    m.scale.z = 1.0
    m.frame_locked = True
    return m

def text_marker(body_frame_id, text="", s=5.0, color="white", alpha=1.0):
    m = colored_marker(color, alpha)
    m.header.frame_id = body_frame_id
    m.ns = "text"
    m.id = 0
    m.type = Marker.TEXT_VIEW_FACING
    m.pose.position.z = 1.0
    m.scale.z = s
    m.text = text
    m.frame_locked = True
    return m

def straight_roadway_marker(roadway, color = "blue", alpha=1.0):
    start_mid = np.array(roadway["start_mid"])
    w = roadway["lane_width"]
    l = roadway["length"]
    ns, nt = rot_unit_vectors(roadway["angle"])

    start_left = start_mid + w*nt
    start_right = start_mid - w*nt
    end_mid = start_mid + l*ns
    end_left = end_mid + w*nt
    end_right = end_mid - w*nt

    m = colored_marker(color, alpha)
    m.header.frame_id = "world"
    m.ns = "roadway"
    m.type = Marker.LINE_LIST
    m.scale.x = 0.5
    m.points = ([Point(start_left[0], start_left[1], 0.),
                 Point(end_left[0], end_left[1], 0.),
                 Point(start_right[0], start_right[1], 0.),
                 Point(end_right[0], end_right[1], 0.)] +
                [Point(start_mid[0]*x + end_mid[0]*(1-x),
                       start_mid[1]*x + end_mid[1]*(1-x),
                       0) for x in np.linspace(0.,1.,50)])
    m.frame_locked = True
    return m

def line_segment_marker(v, w, name, color = "green", alpha=1.0):
    m = colored_marker(color, alpha)
    m.header.frame_id = "world"
    m.ns = name
    m.type = Marker.LINE_STRIP
    m.scale.x = 0.5
    m.points = [Point(v[0], v[1], 0.), Point(w[0], w[1], 0.)]
    m.frame_locked = True
    return m
