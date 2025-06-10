"""Sensory neuron class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import matplotlib.pyplot as plt
import music
import numpy as np
from config.core_models import SimulationParams

SEED = SimulationParams.get_default_seed()
np.random.seed(SEED)


class SensoryNeuron:

    def __init__(self, numNeurons, pos=True, idStart=0, bas_rate=0.0, kp=1.0):

        self._numNeurons = numNeurons
        self._baseline_rate = bas_rate
        self._gain = kp
        self._rate = 0.0
        self._spike = []
        self._pos = pos

        # Set IDs starting from idStart
        id_vect = np.zeros(shape=numNeurons)
        for i in range(numNeurons):
            id_vect[i] = i + idStart
        self._pop = id_vect

        # Output port over which sending spikes
        self._outPort = []
        self.rng = np.random.default_rng(SEED)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def numNeurons(self):
        return self._numNeurons

    @numNeurons.setter
    def numNeurons(self, value):
        self._numNeurons = value

    @property
    def pop(self):
        return self._pop

    @pop.setter
    def pop(self, value):
        self._pop = value

    @property
    def baseline_rate(self):
        return self._baseline_rate

    @baseline_rate.setter
    def baseline_rate(self, value):
        self._baseline_rate = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        self._rate = value

    @property
    def spike(self):
        return self._spike

    @property
    def outPort(self):
        return self._outPort

    def connect(self, port):
        self._outPort = port

    # Update theoretical spike rate based on input signal, and generate spikes
    def update(self, signal, resolution, simTime):

        # Signal according to the sensitivity of the neuron
        if (self.pos and signal < 0) or (not self.pos and signal >= 0):
            signal = 0

        # Tehoretical rate
        self.rate = self.baseline_rate + self.gain * abs(signal)

        # Transform theoretical rate into lambda coefficient (Poisson)
        lmd = self.rate * resolution  # Lamda of Poisson distribution

        nEv = self.rng.poisson(lam=lmd, size=(self.numNeurons))

        for i in range(self.numNeurons):
            if (nEv[i]) > 0:
                self.spike.append([simTime, self.pop[i]])
                if self.outPort:
                    self.outPort.insertEvent(simTime, self.pop[i], music.Index.GLOBAL)
                else:
                    # print("Sensory neuron "+str(self.pop)+" not connected!")
                    pass

    def get_events(self):
        spk = self.spike
        # Sort list of spikes based on time of event
        spk.sort(key=sortFirst)
        # Extract times and neuron ids of events
        if spk:
            ts = np.array(spk)[:, 0]
            evs = np.array(spk)[:, 1]
        else:
            ts = np.nan
            evs = np.nan
        return evs, ts

    # Buffer size in ms
    # NOTE: the time vector is in seconds, therefore buffer_sz needs to be converted
    def computePSTH(self, time, buffer_sz=0.01):
        t_init = time[0]
        t_end = time[len(time) - 1]
        N = len(self.pop)
        evs, ts = self.get_events()
        count, bins = np.histogram(ts, bins=np.arange(t_init, t_end + 1, buffer_sz))
        rate = count / (N * buffer_sz)
        return bins, count, rate

    def plot_rate(self, time, buffer_sz=0.01, title="", ax=None, bar=True, **kwargs):

        t_init = time[0]
        t_end = time[len(time) - 1]

        bins, count, rate = self.computePSTH(time, buffer_sz)
        rate_sm = np.convolve(rate, np.ones(5) / 5, mode="same")

        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots(1)

        if bar:
            ax.bar(bins[:-1], rate, width=bins[1] - bins[0], **kwargs)
            ax.plot(bins[:-1], rate_sm, color="k")
        else:
            ax.plot(bins[:-1], rate_sm, **kwargs)
        ax.set(xlim=(t_init, t_end))
        ax.set_ylabel(title)


#################### Class ends here ####################


def sortFirst(val):
    return val[0]
