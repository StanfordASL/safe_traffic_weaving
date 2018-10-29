# ------------------ /path ----------------------------
# Custom ROS message for path information
# Contains vectors of each element

Header header           # Ros defined header containing timestamp and sequence id
int64 pathNum
float64[] s_m             # Distance along path [m]
float64[] posE_m          # E Position of path point [m]
float64[] posN_m          # N Position of path point [m]
float64[] Psi_rad         # Heading of path at point [rad]
float64[] k_1pm           # Curvature of path at point [1/m]
float64[] grade_rad       # Road grade [rad]
float64[] edge_L_m        # Left lateral deviation bound [m]
float64[] edge_R_m        # Right lateral deviation bound [m]
float64[] Ux_des_mps      # Desired longitudinal speed [m/s]
float64[] Ax_des_mps2     # Desired longitudinal acceleration [m/s^2]
int64     isOpen          # Closed or open path
