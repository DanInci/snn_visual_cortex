import matplotlib
import os

matplotlib.use('Agg')

from brian2 import *


def plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, output_folder=None, file_name='spike_raster_plot'):
    """ Plots the spikes """

    plt.plot(spike_mon_cs.t / ms, spike_mon_cs.i, '.b', label='CS')
    plt.plot(spike_mon_cc.t / ms, len(spike_mon_cs.count) + spike_mon_cc.i, '.r', label='CC')
    plt.plot(spike_mon_sst.t / ms, (len(spike_mon_cs.count) + len(spike_mon_cc.count)) + spike_mon_sst.i, '.g', label='SST')
    plt.plot(spike_mon_pv.t / ms, (len(spike_mon_cs.count) + len(spike_mon_cc.count) + len(spike_mon_sst.count)) + spike_mon_pv.i, '.y', label='PV')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.legend(loc='best')
    plt.title('')

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.savefig('%s/%s.pdf'%(output_folder, file_name), bbox_inches='tight')


def plot_states(state_mon, spike_mon, spike_thld, output_folder=None, file_name='state_plot'):
    """ Plots the variable states for a monitor """

    figure(figsize=(18, 4))
    subplot(1, 2, 1)

    # plot existing membrane potentials
    vs = [v for v in state_mon.record_variables if v.startswith('v_')]
    for v in vs:
        plt.plot(state_mon.t / ms, getattr(state_mon, v)[0], label=v)

    for (t, i) in zip(spike_mon.t, spike_mon.i):
        if i == 0:
            plt.axvline(t / ms, ls='--', c='C1', lw=1)

    plt.axhline(spike_thld / mV / 1000, ls=':', c='C2', lw=3, label='spike thld')
    plt.xlabel('Time (ms)')
    plt.ylabel('potential (V)')
    plt.legend(loc='upper right')

    subplot(1, 2, 2)

    # plot conductance
    gs = [g for g in state_mon.record_variables if g.startswith('g_')]
    for g in gs:
        plt.plot(state_mon.t / ms, getattr(state_mon, g)[0], label=g)

    xlabel('Time (ms)')
    ylabel('Conductance (S)')
    legend(loc='best')

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.savefig('%s/%s.pdf'%(output_folder, file_name), bbox_inches='tight')