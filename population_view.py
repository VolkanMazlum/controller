"""Generic class for a population of neuron."""

__authors__ = "Alberto Antonietti and Cristiano Alessandro"
__copyright__ = "Copyright 2020"
__credits__ = ["Alberto Antonietti and Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"


import nest
import numpy as np
import matplotlib.pyplot as plt
import mpi4py
from settings import Experiment
exp = Experiment()
pathData = exp.pathData + "nest/"

################ TODO: NOT TESTED
class Event:
    def __init__(self, n_id, t):
        self.n_id = n_id
        self.t = t

################ TODO: NOT TESTED
class Events(list):
    def __init__(self, *args):
        if len(args) == 1:
            self._from_list(*args)
        if len(args) == 2:
            self._from_ids_ts(*args)

    def _from_list(self, ev_list):
        super().__init__(ev_list)

    def _from_ids_ts(self, n_ids, ts):
        ev_list = [Event(n_id, t) for (n_id, t) in zip(n_ids, ts)]
        self._from_list(ev_list)

    @property
    def n_ids(self):
        return [e.n_id for e in self]

    @property
    def ts(self):
        return [e.t for e in self]


###################### UTIL ######################

def new_spike_detector(pop,**kwargs):
    spike_detector = nest.Create("spike_recorder")
    nest.SetStatus(spike_detector, params=kwargs)
    nest.Connect(pop, spike_detector)
    return spike_detector


def get_spike_events(spike_detector):
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]

    return evs, ts

def plot_spikes(evs, ts, time_vect, pop=None, title='', ax=None):

    t_init = time_vect[0]
    t_end  = time_vect[ len(time_vect)-1 ]

    no_ax = ax is None
    if no_ax:
        fig, ax = plt.subplots(1)

    ax.scatter(ts, evs, marker='.', s=1)
    ax.set(xlim=(t_init, t_end))
    ax.set_ylabel(title)
    if pop:
        ax.set_ylim([min(pop), max(pop)])


# NOTE: This depends on the protocol (it uses variables like n_trials...)
# NOTE: this assumes a constant rate across the trial
def get_rate(spike_detector, pop, trial_len, n_trials=1):
    rate = nest.GetStatus(spike_detector, keys="n_events")[0] * 1e3 / (trial_len*n_trials)
    rate /= len(pop)
    return rate


def plotPopulation(time_v, pop_pos, pop_neg, reference, time_vecs, legend, styles, title='',buffer_size=15):
    if hasattr(pop_pos, 'total_ts') and hasattr(pop_neg, 'total_ts'):
        #print('c e')
        evs_p = pop_pos.total_evs
        ts_p = pop_pos.total_ts

        evs_n = pop_neg.total_evs
        ts_n = pop_neg.total_ts

        y_p = [ev - int(pop_pos.pop[0].get('global_id')) + 1 for ev in evs_p]
        y_n = [-((ev - int(pop_neg.pop[0].get('global_id'))) + 1) for ev in evs_n]
    else:
        evs_p, ts_p = pop_pos.get_events()
        evs_n, ts_n = pop_neg.get_events()
        y_p =   evs_p - pop_pos.pop[0] + 1
        y_n = -(evs_n - pop_neg.pop[0] + 1)

    
    
    if not reference:
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].scatter(ts_p, y_p, marker='.', s=1,c="r")
        ax[0].scatter(ts_n, y_n, marker='.', s=1)
        ax[0].set_ylabel("raster")
        ax[0].legend()
        pop_pos.plot_rate(time_v, buffer_size, ax=ax[1],color="r")
        pop_neg.plot_rate(time_v, buffer_size, ax=ax[1], title='PSTH (Hz)')
        ax[0].set_title(title)
        ax[0].set_ylim( bottom=-(len(pop_neg.pop)+1), top=len(pop_pos.pop)+1 )
    else:
        fig, ax = plt.subplots(3,1,sharex=True)
        for i, signal in enumerate(reference):
            ax[0].plot(time_vecs[i], signal, styles[i],label=legend[i])
            ax[0].legend()
        ax[1].scatter(ts_p, y_p, marker='.', s=1,c="r")
        ax[1].scatter(ts_n, y_n, marker='.', s=1, color='b')
        ax[1].set_ylabel("raster")
        rate_p = pop_pos.plot_rate(time_v, buffer_size, ax=ax[2],color="r")
        rate_n = pop_neg.plot_rate(time_v, buffer_size, ax=ax[2], title='PSTH (Hz)', color='b')
        ax[1].set_ylim( bottom=-(len(pop_neg.pop)+1), top=len(pop_pos.pop)+1 )
        print('rate net: ', rate_p[-1]- rate_n[-1])
    subplot_labels = ['A', 'B', 'C']
    for i, axs in enumerate(ax):
        axs.text(-0.1, 1.1, subplot_labels[i], transform=axs.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(True)
        ax[i].spines['left'].set_visible(True)

    return fig, ax


############################ POPULATION VIEW #############################
class PopView:
    def __init__(self, pop, time_vect, to_file=False, label=''):
        self.pop = pop
        if to_file==True:
            if label=='':
                raise Exception("To save into file, you need to specify a label")
            #param_file = {"to_file": True, "label":label, "file_extension": "dat"}
            #param_file={"record_to": label + ".dat"}
            param_file = {"record_to": 'ascii', "label": label}
            self.detector = new_spike_detector(pop,**param_file)
        else:
            self.detector = new_spike_detector(pop)

        self.total_n_events = 0
        self.rates_history = []

        self.time_vect = time_vect
        self.trial_len = time_vect[ len(time_vect)-1 ]

    def connect(self, other, rule='one_to_one', w=1.0, d=0.1):
        nest.Connect(self.pop, other.pop, rule, syn_spec={'weight': w, "delay":d})

    def slice(self, start, end=None, step=None):
        return PopView(self.pop[start:end:step])

    def get_events(self):
        return get_spike_events(self.detector)

    def plot_spikes(self, time, boundaries=None, title='', ax=None):
        evs, ts = self.get_events()

        if boundaries is not None:
            i_0, i_1 = boundaries

            selected = Events(
                Event(e.n_id, e.t) for e in Events(evs, ts)
                if self.trial_len*i_0 <= e.t < self.trial_len*i_1
            )
            evs, ts = selected.n_ids, selected.ts

        plot_spikes(evs, ts, time, self.pop, title, ax)

    # Buffer size in ms
    # NOTE: the time vector is in seconds, therefore buffer_sz needs to be converted
    def computePSTH(self, time, buffer_sz=10):
        t_init = time[0]
        t_end  = time[ len(time)-1 ]
        N = len(self.pop)
        if hasattr(self, 'total_evs') and hasattr(self, 'total_ts'):
            evs = self. total_evs
            ts = self.total_ts
        else:
            evs, ts = self.get_events()
        count, bins = np.histogram( ts, bins=np.arange(t_init,t_end+1,buffer_sz) )
        rate = 1000*count/(N*buffer_sz)
        return bins, count, rate


    def plot_rate(self, time, buffer_sz=10, title='', ax=None, bar=True, **kwargs):

        t_init = time[0]
        t_end  = time[ len(time)-1 ]

        
        bins,count,rate = self.computePSTH(time, buffer_sz)
        '''
        rate_sm = np.convolve(rate, np.ones(5)/5,mode='same')
        '''
        rate_padded = np.pad(rate, pad_width=2, mode='reflect') 
        rate_sm = np.convolve(rate_padded, np.ones(5) / 5, mode='valid')
        
    
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots(1)

        if bar:
            ax.bar(bins[:-1], rate, width=bins[1]-bins[0],**kwargs)
            ax.plot(bins[:-1],rate_sm,color='k')
        else:
            ax.plot(bins[:-1],rate_sm,**kwargs)
        ax.set(xlim=(t_init, t_end))
        ax.set_ylabel(title)
        ax.set_xlabel('Time [ms]')

        return rate

    ########## ACROSS TRIALS STUFF ##########

    # NOTE: only for constant signals
    def get_rate(self, n_trials=1):
        return get_rate(self.detector, self.pop, self.trial_len, n_trials)

    def reset_per_trial_rate(self):
        self.total_n_events = 0
        self.rates_history = []

    def get_per_trial_rate(self, trial_i=None):
        if trial_i is not None:
            n_ids, ts = self.get_events()
            events = Events(n_ids, ts)

            trial_events = Events(
                Event(e.n_id, e.t) for e in events
                if self.trial_len*trial_i <= e.t < self.trial_len*(trial_i+1)
            )
            n_events = len(trial_events)
        else:
            print("Warning: deprecated, pass trial_i explicitly")
            n_events = nest.GetStatus(self.detector, keys="n_events")[0]

            n_events -= self.total_n_events
            self.total_n_events += n_events

        rate = n_events * 1e3 / self.trial_len
        rate /= len(self.pop)

        self.rates_history.append(rate)
        return rate

    def plot_per_trial_rates(self, title='', ax=None):
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots(1)

        ax.plot(self.rates_history)
        ax.set_ylabel(title)

        if no_ax:
            plt.show()
    
    def gather_data(self, senders, times):
        self.total_evs = senders
        self.total_ts = times
        


