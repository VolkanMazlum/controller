__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt

# Save pattern into file
def savePattern(pattern, file_bas):

    nj = pattern.shape[1]

    for i in range(nj):
        cmd_file = file_bas + "_" + str(i) + ".dat"
        a_file = open(cmd_file, "w")
        np.savetxt( a_file, pattern[:,i] )
        a_file.close()


# Plot positive and negative population
def plotPopulation(time_v, pop_pos, pop_neg, title='',buffer_size=15):
    evs_p, ts_p = pop_pos.get_events()
    evs_n, ts_n = pop_neg.get_events()

    y_p =   evs_p - pop_pos.pop[0] + 1
    y_n = -(evs_n - pop_neg.pop[0] + 1)

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].scatter(ts_p, y_p, marker='.', s=1,c="r")
    ax[0].scatter(ts_n, y_n, marker='.', s=1)
    ax[0].set_ylabel("raster")
    pop_pos.plot_rate(time_v, buffer_size, ax=ax[1],color="r")
    pop_neg.plot_rate(time_v, buffer_size, ax=ax[1], title='PSTH (Hz)')
    ax[0].set_title(title)
    ax[0].set_ylim( bottom=-(len(pop_neg.pop)+1), top=len(pop_pos.pop)+1 )

    return fig, ax

def AddPause(signal, pause_len, res):    
    # Add a pause at the end of the signal pattern
    signal_list = list(signal)
    signal_list.extend([0]*int(pause_len/res))

    return np.array(signal_list)
