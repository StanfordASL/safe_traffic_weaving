# See https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates

from __future__ import division
import numpy as np
from numpy import sin, cos, pi, sqrt, hypot, cbrt, arctan, arctan2

# CONSTANTS
D2R = pi/180
R2D = 180/pi
a = 6378137             # ellipsoid's semi-major axis
b = 6356752.3           # ellipsoid's semi-minor axis
e2 = 1 - b*b/(a*a)      # ellipsoid's eccentricity^2
ep2 = e2/(1-e2)         # used in ecef2lla
E2 = e2*a*a             # used in ecef2lla

def lla2ecef(lla):
    clat = cos( lla[0] * D2R )
    clon = cos( lla[1] * D2R )
    slat = sin( lla[0] * D2R )
    slon = sin( lla[1] * D2R )
    r0   = a / ( sqrt( 1.0 - e2 * slat * slat ) );
    return np.array([
        ( lla[2] + r0 ) * clat * clon,         # x-coord
        ( lla[2] + r0 ) * clat * slon,         # y-coord
        ( lla[2] + r0 * ( 1.0 - e2 ) ) * slat   # z-coord
    ])

def ecef2lla(ecef):
    X = ecef[0]
    Y = ecef[1]
    Z = ecef[2]

    r = hypot(X, Y)
    F = 54*b*b*Z*Z
    G = r*r + (1 - e2)*Z*Z - e2*E2
    C = e2*e2*F*r*r/(G*G*G)
    S = cbrt(1 + C + sqrt(C*C + 2*C))
    T = S + 1/S + 1
    P = F/(3*T*T*G*G)
    Q = sqrt(1 + 2*e2*e2*P)
    r0 = -P*e2*r/(1+Q) + sqrt((1+1/Q)*a*a/2 - P*(1-e2)*Z*Z/(Q*(1+Q)) - P*r*r/2)
    U = hypot(r - e2*r0, Z)
    V = hypot(r - e2*r0, Z*b/a)
    Z0 = b*b*Z/(a*V)
    return np.array([
        arctan((Z + ep2*Z0)/r)*R2D,
        arctan2(Y, X)*R2D,
        U*(1 - b*b/(a*V))
    ])

def R_ecef2enu( llaRef ):
    clatRef = cos( llaRef[0] * D2R )
    clonRef = cos( llaRef[1] * D2R )
    slatRef = sin( llaRef[0] * D2R )
    slonRef = sin( llaRef[1] * D2R )
    return np.array([
        [          -slonRef,             clonRef,        0],
        [-slatRef * clonRef,  -slatRef * slonRef,  clatRef],
        [ clatRef * clonRef,   clatRef * slonRef,  slatRef]
    ])    

def lla2enu( llaRef, lla ):
    ecefRef, R = lla2ecef(llaRef), R_ecef2enu(llaRef)
    ecef = lla2ecef(lla)
    enu = np.dot(R, ecef - ecefRef)
    return enu

def enu2lla( llaRef, enu ):
    ecefRef, R = lla2ecef(llaRef), R_ecef2enu(llaRef)
    ecef = np.linalg.solve(R, enu) + ecefRef
    lla = ecef2lla(ecef)
    return lla
