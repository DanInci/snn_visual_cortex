from brian2 import ms, mV
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')

from matplotlib import pyplot as plt


index_to_ntype_dict = {
    0: 'CS',
    1: 'CC',
    2: 'SST',
    3: 'PV'
}


def plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, from_t=None, to_t=None,
                output_folder=None, file_name='spike_raster_plot'):
    """ Plots the spikes """

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.scatter(spike_mon_cs.t / ms, spike_mon_cs.i, s=0.2, color='b', label='CS')
    ax1.scatter(spike_mon_cc.t / ms, len(spike_mon_cs.count) + spike_mon_cc.i, s=0.2, color='r', label='CC')
    ax1.scatter(spike_mon_sst.t / ms, (len(spike_mon_cs.count) + len(spike_mon_cc.count)) + spike_mon_sst.i, s=0.2, color='g', label='SST')
    ax1.scatter(spike_mon_pv.t / ms, (len(spike_mon_cs.count) + len(spike_mon_cc.count) + len(spike_mon_sst.count)) + spike_mon_pv.i, s=0.2, color='y', label='PV')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron index')
    ax1.legend(loc='best')
    ax1.set_title('Spike Raster Plot')

    ax1.set_xlim(left=from_t / ms, right=to_t / ms)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')


def plot_states(state_mon, spike_mon, spike_thld,
                plot_only_from_equilibrium=False, from_t=None, to_t=None,
                output_folder=None, file_name='state_plot', record=0):
    """ Plots the variable states for a monitor """

    fig, axs = plt.subplots(1, 2, figsize=(18, 4))

    # plot existing membrane potentials
    vs = [v for v in state_mon.record_variables if v.startswith('v')]
    for v in vs:
        axs[0].plot(state_mon.t / ms, getattr(state_mon, v)[record], label=v)

    for (t, i) in zip(spike_mon.t, spike_mon.i):
        if i == 0:
            axs[0].axvline(t / ms, ls='--', c='C1', lw=1)

    axs[0].axhline(spike_thld / mV / 1000, ls=':', c='C2', lw=3, label='spike thld')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('potential (V)')
    axs[0].legend(loc='upper right')

    if plot_only_from_equilibrium:
        axs[0].set_xlim(left=from_t / ms, right=to_t / ms)

    # plot conductance
    gs = [g for g in state_mon.record_variables if g.startswith('g')]
    for g in gs:
        axs[1].plot(state_mon.t / ms, getattr(state_mon, g)[record], label=g)

    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Conductance (S)')
    axs[1].legend(loc='best')

    if plot_only_from_equilibrium:
        axs[1].set_xlim(left=from_t / ms, right=to_t / ms)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')


def plot_firing_rate_histograms(firing_rates, no_bins, output_folder=None, file_name='firing_rate_histograms'):
    columns = 2
    rows = int(len(firing_rates) / columns)

    fig, axs = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows))

    for (ntype_index, firing_rate_i) in enumerate(firing_rates):
        row_idx = int(ntype_index / columns)
        col_idx = ntype_index % columns

        # plot histogram of neuron group
        axs[row_idx][col_idx].hist(firing_rate_i, bins=no_bins)
        axs[row_idx][col_idx].axis(ymin=0)
        axs[row_idx][col_idx].set_title(f'Neuron group {index_to_ntype_dict[ntype_index]}', fontsize=10)
        axs[row_idx][col_idx].set_xlabel("Firing rate [Hz]", fontsize=10)
        axs[row_idx][col_idx].set_ylabel("Frequency", fontsize=10)
        axs[row_idx][col_idx].tick_params(axis='both', which='major', labelsize=10)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')


def plot_isi_histograms(interspike_intervals, no_bins, autocorr=None, output_folder=None, file_name='isi_histograms'):
    columns = 2
    rows = len(interspike_intervals)

    fig, axs = plt.subplots(rows, columns, figsize=(6*columns, 6*rows))

    for (ntype_index, interspike_intervals_i) in enumerate(interspike_intervals):
        row_idx = ntype_index

        if autocorr:
            acorr_struct = autocorr[ntype_index]

        if acorr_struct:
            xaxis = acorr_struct["xaxis"]
            acorr = acorr_struct["acorr"]
            minimum = acorr_struct["minimum"]
            label_minimum = f"maxISI {str(np.round(xaxis[minimum], 4))} s"

        # plot histogram of neuron group
        n, bins, patches = axs[row_idx][0].hist(interspike_intervals_i, bins=no_bins)
        axs[row_idx][0].axis(ymin=0)
        axs[row_idx][0].set_title(f'Neuron group {index_to_ntype_dict[ntype_index]}', fontsize=10)
        axs[row_idx][0].set_xlabel("ISI [s]", fontsize=10)
        axs[row_idx][0].set_ylabel("Frequency", fontsize=10)
        axs[row_idx][0].tick_params(axis='both', which='major', labelsize=10)
        if acorr_struct and minimum:
            axs[row_idx][0].vlines(xaxis[minimum], 0, np.max(n), label=label_minimum, color='red')
            axs[row_idx][0].legend()

        # plot auto-correlation function of isi for neuron group
        if acorr_struct:
            axs[row_idx][1].plot(xaxis, acorr, c='k')
            axs[row_idx][1].set_title(f'Neuron group {index_to_ntype_dict[ntype_index]}', fontsize=10)
            axs[row_idx][1].set_xlabel("time lag [s]")
            axs[row_idx][1].set_ylabel("norm. autocorr.")
            if minimum:
                axs[row_idx][1].vlines(xaxis[minimum], 0, 1, label=label_minimum, color='red')
                axs[row_idx][1].legend()

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')