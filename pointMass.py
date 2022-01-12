"""Point mass class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
from body import Body

class PointMass(Body):

    def __init__(self, IC_pos=np.array([0.0,0.0]),
                       IC_vel=np.array([0.0,0.0]),
                       mass = 1.0 ):

        if IC_pos.shape!=IC_vel.shape:
            raise Exception("Position and velocity need to have the same format")

        self.mass = mass
        self.pos  = IC_pos
        self.vel  = IC_vel

    @property
    def mass(self):
        return self._mass

    # State variable setter
    @mass.setter
    def mass(self, value):
        self._mass = value

    # Integrate the body over dt
    def integrateTimeStep(self, u, dt):
        if u.shape!=self._pos.shape or u.shape!=self._vel.shape:
            raise Exception("Wrong format")
        self._vel = self._vel + u/self._mass * dt
        self._pos = self._pos + self._vel * dt

    # In a point mass the inverse kinematics is trivial: the position of the
    # state variable equals that of the external variable (i.e. position of the
    # point mass in the cartesian space)
    def inverseKin(self,pos_external):
        # positions, velocities and accelerations are expected to be N-by-nV arrays,
        # where N is the number of timesteps and nV is the number of variables
        if pos_external.ndim>1:
            if pos_external.shape[1]!=self._pos.shape[0]:
                raise Exception("Wrong value format")
        elif pos_external.shape!=self._pos.shape:
            raise Exception("Wrong value format")

        return pos_external

    # Similarly to the inverse kinematics, the forward kinematics is trivial
    def forwardKin(self,pos):
        if pos.ndim>1:
            if pos.shape[1]!=self._pos.shape[0]:
                raise Exception("Wrong value format")
        elif pos.shape!=self._pos.shape:
            raise Exception("Wrong value format")

        return pos

    # In a point mass the inverse dynamics is the simple multiplication of the
    # acceleration of the point by its mass
    def inverseDyn(self,pos,vel,acc):
        # positions, velocities and accelerations are expected to be N-by-nV arrays,
        # where N is the number of timesteps and nV is the number of variables
        if acc.ndim>1:
            if acc.shape[1]!=self._pos.shape[0]:
                raise Exception("Wrong value format")
        elif acc.shape!=self._pos.shape:
            raise Exception("Wrong value format")

        return self._mass*acc

    # In a point mass the Jacobian is the identity matrix (independent of position)
    def jacobian(self,position):
        return np.identity(self.numVariables())


# TEST
if __name__ == '__main__':

    m   = 5.0
    #ICp = np.array([2.0, 2.0, 3.0])
    #ICv = np.array([3.0, 3.0, 4.0])
    ICp = np.array([2.0, 2.0])
    ICv = np.array([3.0, 3.0])

    pm = PointMass(mass=m,IC_pos=ICp,IC_vel=ICv)

    print("Mass: "+str(pm.mass))
    print("Position: "+str(pm.pos))
    print("Velocity: "+str(pm.vel))

    #ext_pos = np.array([7.0,10.0])         # Correct
    #ext_pos = np.array([7.0,10.0,3.0])     # Wrong
    #ext_pos = 7.0                          # Wrong
    #ext_pos = np.ones((10,2))              # Correct
    #ext_pos = np.ones((10,3))              # Wrong
    #print( pm.inverseKin(ext_pos) )

    #acc = np.array([3.0,4.0])               # Correct
    #acc = np.array([3.0,4.0,7.0])           # Wrong
    #acc = 7.0                               # Wrong
    #acc = np.ones((10,2))                   # Correct
    #acc = np.ones((10,3))                   # Wrong
    #print( pm.inverseDyn(acc) )

    print("Number of variables: "+str(pm.numVariables()))

    # Integration
    dt = 0.01
    #u  = np.array([5.0,3.0])
    u  = np.array([0.0,0.0])
    pm.integrateTimeStep(u,dt)
    print("New position: "+str(pm.pos))
    print("New velocity: "+str(pm.vel))
