"""Perturbations"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"


import numpy as np
import matplotlib.pyplot as plt


def curledForceField(vel, angle, k):
    # Rotate the velocity vector of angle (degrees)
    ang_r = angle * np.pi/180
    R = np.matrix( [[np.cos(ang_r), -np.sin(ang_r)],
                    [np.sin(ang_r),  np.cos(ang_r)]] )
    # Multiply the rotated vector by a coefficient k
    force = k * np.matmul(R,vel)
    return force


# TEST
if __name__ == '__main__':

    v = np.array([5,5])

    k    = 3.0
    ang  = 90

    f = curledForceField(v, ang, k)
    
    print("force: "+str(f))

    v_mag = np.linalg.norm(v)
    f_mag = np.linalg.norm(f)

    print("Velocity magnitude: "+str(v_mag))
    print("Force magnitude: "+str(f_mag))
    print("Desired force magnitude: "+str(k*v_mag))
