from brian2 import ms, mV
import numpy as np
import matplotlib
import os

from matplotlib.lines import Line2D

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

    N_cs = len(spike_mon_cs.count)
    N_cc = len(spike_mon_cc.count)
    N_sst = len(spike_mon_sst.count)
    N_pv = len(spike_mon_pv.count)
    N = N_cs + N_cc + N_sst + N_pv

    if len(spike_mon_cs.i[:, np.newaxis]) > 0:
        ax1.eventplot(spike_mon_cs.i[:, np.newaxis], lineoffsets=spike_mon_cs.t / ms, orientation='vertical', colors='b', linewidths=2)
        # ax1.axhline(N_cs - 1/2, lw=0.5, color='k')

    if len((N_cs + spike_mon_cc.i)[:, np.newaxis]) > 0:
        ax1.eventplot((N_cs + spike_mon_cc.i)[:, np.newaxis], lineoffsets=spike_mon_cc.t / ms, orientation='vertical', colors='r', linewidths=2)
        # ax1.axhline(N_cs + N_cc - 1/2, lw=0.5, color='k')

    if len(((N_cs + N_cc) + spike_mon_sst.i)[:, np.newaxis]):
        ax1.eventplot(((N_cs + N_cc) + spike_mon_sst.i)[:, np.newaxis], lineoffsets=spike_mon_sst.t / ms, orientation='vertical', colors='g', linewidths=2)
        # ax1.axhline(N_cs + N_cc + N_sst - 1/2, lw=0.5, color='k')

    if len(((N_cs + N_cc + N_sst) + spike_mon_pv.i)[:, np.newaxis]) > 0:
        ax1.eventplot(((N_cs + N_cc + N_sst) + spike_mon_pv.i)[:, np.newaxis], lineoffsets=spike_mon_pv.t / ms, orientation='vertical', colors='y', linewidths=2)

    custom_handles = [Line2D([0], [0], color='y', lw=1, label='PV'),
                      Line2D([0], [0], color='g', lw=1, label='SST'),
                      Line2D([0], [0], color='r', lw=1, label='CS'),
                      Line2D([0], [0], color='b', lw=1, label='CC')]

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron index')
    ax1.legend(handles=custom_handles, loc='best')
    ax1.set_title('Spike Raster Plot')

    ax1.set_xlim(left=from_t / ms, right=to_t / ms)
    ax1.set_ylim(bottom=-1 / 2, top=N - 1 / 2)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')

    plt.close(fig)


def plot_states(state_mon, spike_mon, spike_thld,
                from_t=None, to_t=None,
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
    axs[0].set_xlim(left=from_t / ms, right=to_t / ms)

    # plot conductance
    gs = [g for g in state_mon.record_variables if g.startswith('g')]
    for g in gs:
        axs[1].plot(state_mon.t / ms, getattr(state_mon, g)[record], label=g)

    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Conductance (S)')
    axs[1].legend(loc='best')
    axs[1].set_xlim(left=from_t / ms, right=to_t / ms)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')

    plt.close(fig)


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

    plt.close(fig)


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

    plt.close(fig)


def plot_selectivity_comparison(agg_results_vector, output_folder=None, file_name='selectivity_comparison'):
    """ Plots the orientation and direction selectivity for all neuron groups"""

    fig, axs = plt.subplots(3, 1, figsize=(6 * len(agg_results_vector), 18))
    bar_width = 0.8 / len(agg_results_vector)
    x_offset = 0.05
    x = np.arange(len(agg_results_vector)) / len(agg_results_vector) + x_offset
    ticks = x + bar_width / 4 + x_offset
    labels = [f'SST->Soma {agg_results["pSST_CS_soma"]}CS/{agg_results["pSST_CC_soma"]}CC' for agg_results in agg_results_vector]

    orientation_s_cs = [agg_results["output_selectivity_cs"]["orientation"] for agg_results in agg_results_vector]
    orientation_s_cc = [agg_results["output_selectivity_cc"]["orientation"] for agg_results in agg_results_vector]
    orientation_s_sst = [agg_results["output_selectivity_sst"]["orientation"] for agg_results in agg_results_vector]
    orientation_s_pv = [agg_results["output_selectivity_pv"]["orientation"] for agg_results in agg_results_vector]

    # plot orientation selectivity
    axs[0].bar(x, orientation_s_cs, bar_width / 4, label="CS", color='b')
    axs[0].bar(x + bar_width / 4, orientation_s_cc, bar_width / 4, label="CC", color='r')
    axs[0].bar(x + bar_width / 2, orientation_s_sst, bar_width / 4, label="SST", color='g')
    axs[0].bar(x + bar_width * 3 / 4, orientation_s_pv, bar_width / 4, label="PV", color='y')
    axs[0].set_ylabel('Orientation selectivity')
    axs[0].set_title('Orientation selectivity')
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(labels)
    axs[0].legend(loc='best')

    orientation_s_paper_cs = [agg_results["output_selectivity_cs"]["orientation_paper"] for agg_results in agg_results_vector]
    orientation_s_paper_cc = [agg_results["output_selectivity_cc"]["orientation_paper"] for agg_results in agg_results_vector]
    orientation_s_paper_sst = [agg_results["output_selectivity_sst"]["orientation_paper"] for agg_results in agg_results_vector]
    orientation_s_paper_pv = [agg_results["output_selectivity_pv"]["orientation_paper"] for agg_results in agg_results_vector]

    # plot orientation selectivity
    axs[1].bar(x, orientation_s_paper_cs, bar_width / 4, label="CS", color='b')
    axs[1].bar(x + bar_width / 4, orientation_s_paper_cc, bar_width / 4, label="CC", color='r')
    axs[1].bar(x + bar_width / 2, orientation_s_paper_sst, bar_width / 4, label="SST", color='g')
    axs[1].bar(x + bar_width * 3 / 4, orientation_s_paper_pv, bar_width / 4, label="PV", color='y')
    axs[1].set_ylabel('Orientation selectivity (paper)')
    axs[1].set_title('Orientation selectivity (paper)')
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(labels)
    axs[1].legend(loc='best')

    direction_s_cs = [agg_results["output_selectivity_cs"]["direction"] for agg_results in agg_results_vector]
    direction_s_cc = [agg_results["output_selectivity_cc"]["direction"] for agg_results in agg_results_vector]
    direction_s_sst = [agg_results["output_selectivity_sst"]["direction"] for agg_results in agg_results_vector]
    direction_s_pv = [agg_results["output_selectivity_pv"]["direction"] for agg_results in agg_results_vector]

    # plot direction selectivity
    axs[2].bar(x, direction_s_cs, bar_width / 4, label="CS", color='b')
    axs[2].bar(x + bar_width / 4, direction_s_cc, bar_width / 4, label="CC", color='r')
    axs[2].bar(x + bar_width / 2, direction_s_sst, bar_width / 4, label="SST", color='g')
    axs[2].bar(x + bar_width * 3 / 4, direction_s_pv, bar_width / 4, label="PV", color='y')
    axs[2].set_ylabel('Direction selectivity')
    axs[2].set_title('Direction selectivity')
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels(labels)
    axs[2].legend(loc='best')

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')

    plt.close(fig)


def visualise_synapse_group_connectivity(ax, S, S_reverse):
    if S:
        Ns = len(S.source)
        Nt = len(S.target)
        ax.set_xlim(-1, Ns)
        ax.set_ylim(-1, Nt)
        ax.plot(S.i, S.j, '>b')

    if S_reverse:
        ax.plot(S_reverse.j, S_reverse.i, '<g')

    ax.set_xlabel('Source neuron index')
    ax.set_ylabel('Target neuron index')


def visualise_SST_connectivity(axs, conn):
    axs[0].set_title('SST <=> SST')
    visualise_synapse_group_connectivity(axs[0], conn.get("SST_SST"), conn.get("SST_SST"))

    axs[1].set_title('SST <=> PV')
    visualise_synapse_group_connectivity(axs[1], conn.get("SST_PV"), conn.get("PV_SST"))

    axs[2].set_title('SST <=> CS Soma')
    visualise_synapse_group_connectivity(axs[2], conn.get("SST_CSsoma"), conn.get("CSsoma_SST"))

    axs[3].set_title('SST => CS Dendrite')
    visualise_synapse_group_connectivity(axs[3], conn.get("SST_CSdendrite"), None)

    axs[4].set_title('SST <=> CC Soma')
    visualise_synapse_group_connectivity(axs[4], conn.get("SST_CCsoma"), conn.get("CCsoma_SST"))

    axs[5].set_title('SST => CC Dendrite')
    visualise_synapse_group_connectivity(axs[5], conn.get("SST_CCdendrite"), None)


def visualise_PV_connectivity(axs, conn):
    axs[0].set_title('PV <=> SST')
    visualise_synapse_group_connectivity(axs[0], conn.get("PV_SST"), conn.get("SST_PV"))

    axs[1].set_title('PV <=> PV')
    visualise_synapse_group_connectivity(axs[1], conn.get("PV_PV"), conn.get("PV_PV"))

    axs[2].set_title('PV <=> CS Soma')
    visualise_synapse_group_connectivity(axs[2], conn.get("PV_CSsoma"), conn.get("CSsoma_PV"))

    axs[3].set_title('PV =/= CS Dendrite')
    #     visualise_synapse_group_connectivity(axs[3], None, None)

    axs[4].set_title('PV <=> CC Soma')
    visualise_synapse_group_connectivity(axs[4], conn.get("PV_CCsoma"), conn.get("CCsoma_PV"))

    axs[5].set_title('PV =/= CC Dendrite')
    #     visualise_synapse_group_connectivity(axs[5], None, None)


def visualise_CS_connectivity(axs, conn):
    axs[0].set_title('CS Soma <=> SST')
    visualise_synapse_group_connectivity(axs[0], conn.get("CSsoma_SST"), conn.get("SST_CSsoma"))

    axs[1].set_title('CS Soma <=> PV')
    visualise_synapse_group_connectivity(axs[1], conn.get("CSsoma_PV"), conn.get("PV_CSsoma"))

    axs[2].set_title('CS Soma <=> CS Soma')
    visualise_synapse_group_connectivity(axs[2], conn.get("CSsoma_CSsoma"), conn.get("CSsoma_CSsoma"))

    axs[3].set_title('CS Soma =/= CS Dendrite')
    #     visualise_synapse_group_connectivity(axs[3], None, None)

    axs[4].set_title('CS Soma =/= CC Soma')
    #     visualise_synapse_group_connectivity(axs[4], None, None)

    axs[5].set_title('CS Soma =/= CC Dendrite')
    #     visualise_synapse_group_connectivity(axs[5], None, None)


def visualise_CC_connectivity(axs, conn):
    axs[0].set_title('CC Soma <=> SST')
    visualise_synapse_group_connectivity(axs[0], conn.get("CCsoma_SST"), conn.get("SST_CCsoma"))

    axs[1].set_title('CC Soma <=> PV')
    visualise_synapse_group_connectivity(axs[1], conn.get("CCsoma_PV"), conn.get("PV_CCsoma"))

    axs[2].set_title('CC Soma <=> CS Soma')
    visualise_synapse_group_connectivity(axs[2], conn.get("CCsoma_CCsoma"), conn.get("CCsoma_CCsoma"))

    axs[3].set_title('CC Soma =/= CS Dendrite')
    #     visualise_synapse_group_connectivity(axs[3], None, None)

    axs[4].set_title('CC Soma =/= CC Soma')
    #     visualise_synapse_group_connectivity(axs[4], None, None)

    axs[5].set_title('CC Soma =/= CC Dendrite')
    #     visualise_synapse_group_connectivity(axs[5], None, None)


def plot_neuron_connectivity(connections, output_folder=None, file_name='neuron_connectivity'):
    fig, axs = plt.subplots(4, 6, figsize=(24, 36))

    visualise_SST_connectivity(axs[0], connections)
    visualise_PV_connectivity(axs[1], connections)
    visualise_CS_connectivity(axs[2], connections)
    visualise_CC_connectivity(axs[3], connections)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig.savefig('%s/%s.pdf' % (output_folder, file_name), bbox_inches='tight')

    plt.close(fig)
