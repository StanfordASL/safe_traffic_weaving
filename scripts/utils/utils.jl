# @rosimport visualization_msgs.msg: Marker
# rostypegen()
using Colors
import visualization_msgs.msg: Marker

# const CAR_X = 4.22100019455    # VTD car
# const CAR_Y = 1.76199996471
# const CAR_dX = 1.3654999733

const CAR_X  = 3.79    # X1 equivalent
const CAR_Y  = 1.87
const CAR_dX = 0.095

function car_marker(body_frame_id, color=(255,0,0), alpha=1.0)
    color isa String && (color = parse(Colorant, color))
    color isa RGB && (color = (color.r.i, color.g.i, color.b.i))
    eltype(color) isa Integer && (color = (c/255 for c in color))

    m = Marker()
    m.header.frame_id = body_frame_id
    m.ns = "rectangle"
    m.id = 0
    m.type = Marker[:CUBE]
    m.pose.position.x = CAR_dX
    m.pose.orientation.w = 1.0
    m.scale.x = CAR_X
    m.scale.y = CAR_Y
    m.scale.z = 1.0
    m.color.r = color[1]/255
    m.color.g = color[2]/255
    m.color.b = color[3]/255
    m.color.a = alpha
    m.frame_locked = true
    m
end

function text_marker(body_frame_id, color=(255,0,0), s=1.0, alpha=1.0)
    color isa String && (color = parse(Colorant, color))
    color isa RGB && (color = (color.r.i, color.g.i, color.b.i))
    eltype(color) isa Integer && (color = (c/255 for c in color))

    m = Marker()
    m.header.frame_id = body_frame_id
    m.ns = "text"
    m.id = 1
    m.type = Marker[:TEXT_VIEW_FACING]
    m.pose.orientation.w = 1.0
    m.pose.position.z = 1.0
    m.scale.z = s
    m.color.r = color[1]/255
    m.color.g = color[2]/255
    m.color.b = color[3]/255
    m.color.a = alpha
    m.frame_locked = true
    m
end
