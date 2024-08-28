"""Util to compute trajectories"""

__authors__ = "Alberto Antonietti, Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Alberto Antonietti, Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"


import numpy as np
import matplotlib.pyplot as plt


def minimumJerk(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =   6*(x_des-x_init)/np.power(T_max,5)
    b = -15*(x_des-x_init)/np.power(T_max,4)
    c =  10*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)
    e =  np.zeros(x_init.shape)
    g =  x_init

    pol = np.array([a,b,c,d,e,g])
    pp  = a*np.power(tmspn,5) + b*np.power(tmspn,4) + c*np.power(tmspn,3) + g

    return pp, pol


def minimumJerk_ddt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  120*(x_des-x_init)/np.power(T_max,5)
    b = -180*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(tmspn,3) + b*np.power(tmspn,2) + c*np.power(tmspn,1) + d

    return pp*1000*1000, pol


# Get the extremes of the second derivative of a min-jerk trajectory
def minJerk_ddt_minmax(x_init, x_des, timespan):

    T_max   = timespan[ len(timespan)-1 ]
    t1      = T_max/2 - T_max/720 * np.sqrt(43200)
    t2      = T_max/2 + T_max/720 * np.sqrt(43200)
    pp, pol = minimumJerk_ddt(x_init, x_des, timespan)

    ext    = np.empty(shape=(2,x_init.size))
    ext[:] = 0.0
    t      = np.empty(shape=(2,x_init.size))
    t[:]   = 0.0

    for i in range(x_init.size):
        if (x_init[i]!=x_des[i]):
            tmp      = np.polyval( pol[:,i],[t1,t2] )
            ext[:,i] = np.reshape( tmp,(1,2) )
            t[:,i]   = np.reshape( [t1,t2],(1,2) )

    return t, ext


def minimumJerk_dddt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  360*(x_des-x_init)/np.power(T_max,5)
    b = -360*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)

    pol = np.array([a,b,c])
    pp  = a*np.power(tmspn,2) + b*np.power(tmspn,1) + c

    return pp, pol


def polar2cartesian(radius, angle):
    ang_rad = angle * np.pi/180
    x = eps_zero( radius*np.cos(ang_rad),1e-10 )
    y = eps_zero( radius*np.sin(ang_rad),1e-10 )
    return np.array([x, y])


def eps_zero(num,eps):
    if (abs(num)<eps):
        num=0.0
    return num


# TEST
if __name__ == '__main__':

    T_max     = 1000.0
    time_vect = np.arange(0, T_max)
    time_vect = time_vect.reshape(time_vect.size,1)

    rad    = 10.0
    x_init = np.array([0.0, 0.0])
    x_des  = polar2cartesian(rad, 75.0)

    trj, pol = minimumJerk(x_init, x_des, time_vect)

    fig, axs = plt.subplots(2, 1, sharex='col')
    axs[0].plot( time_vect, trj[:,0] )
    axs[1].plot( time_vect, trj[:,1] )

    ###

    T_max     = 1000.0
    time_vect = np.arange(0, T_max)
    time_vect = time_vect.reshape(time_vect.size,1)

    rad    = 10.0
    x_init = np.array([0.0, 0.0])

    fig, axs = plt.subplots(2, 1, sharex='col')

    for i in np.arange(0,90+1,30):

        x_des = polar2cartesian(rad, i)

        trj, pol       = minimumJerk_ddt(x_init, x_des, time_vect)
        ext_t, ext_val = minJerk_ddt_minmax(x_init, x_des, time_vect)

        axs[0].plot( time_vect, trj[:,0], label=str(i) )
        axs[0].plot( ext_t, ext_val[:,0], 'o' )

        axs[1].plot( time_vect, trj[:,1], label=str(i) )
        axs[1].plot( ext_t, ext_val[:,1], 'o' )

        # Comparison with sine
        axs[0].plot( time_vect, np.max(ext_val[:,0])*np.sin((2*np.pi*time_vect/T_max)), linestyle=':', label=str(i) )
        axs[1].plot( time_vect, np.max(ext_val[:,1])*np.sin((2*np.pi*time_vect/T_max)), linestyle=':', label=str(i) )

    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    plt.show()
