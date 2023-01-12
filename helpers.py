from brian2 import *
import pandas as pd
import json
import os

from scipy.signal import argrelextrema


# TODO see how to reference this from equations
@check_units(x=volt, result=1)
def sigmoid(x):
    ### Sigmoid function params
    E_d = -38 * mV  # position control of threshold
    D_d = 6 * mV  # sharpness control of threshold

    return 1/(1+exp(-(-x-E_d)/D_d))


def bin(spiketime, dt):
    if len(spiketime) == 0:
        return []

    spiketime = np.array(spiketime)-spiketime[0]
    indexes = np.round(spiketime/dt).astype(int)
    A_t = np.zeros(indexes[-1]+1)
    for i in indexes:
        A_t[i] += 1

    return A_t


def count_spikes_for_neuron_type(spike_mon, from_t=None, to_t=None):
    """
    Computes spike count by neuron for all neurons monitored by a spike monitor between given interval of time
    """

    spike_counts = {}

    for index in spike_mon.spike_trains():
        a = pd.Series(spike_mon.spike_trains()[index] / second)

        if from_t is not None and to_t is not None:
            a = a.loc[lambda x: (x >= from_t) & (x <= to_t)]

        spike_counts[index] = a.size

    return pd.Series(spike_counts)


@check_units(sim_duration=ms)
def compute_firing_rate_for_neuron_type(spike_mon, from_t, to_t):
    """
    Computes firing rate by neuron for all neurons monitored by a spike monitor between given interval of time
    """

    spikes_for_i = count_spikes_for_neuron_type(spike_mon, from_t, to_t)
    duration = to_t - from_t

    return spikes_for_i / duration


def compute_interspike_intervals(spike_mon, from_t, to_t):
    """
    Computes inter-spike intervals by neuron for all neurons monitored by a spike monitor between given interval of time
    """

    by_neuron = []

    for neuron_index in spike_mon.spike_trains():
        spikes_for_neuron = pd.Series(spike_mon.spike_trains()[neuron_index] / second)
        filterd_by_period = spikes_for_neuron.loc[lambda x: (x >= from_t) & (x <= to_t)]

        interspike_intervals = np.diff(filterd_by_period)
        by_neuron.append(interspike_intervals)

    return by_neuron


def compute_autocorr(spike_intervals):
    """
    Computes auto-correlation function on the inter-spike intervals
    """

    if len(spike_intervals) == 0:
        return None, None

    autocorr = plt.acorr(spike_intervals, normed=True, maxlags=None)
    right_xaxis = autocorr[0][int(len(autocorr[1]) / 2):]
    right_acorr = autocorr[1][int(len(autocorr[1]) / 2):]
    return right_xaxis, right_acorr


def find_minimum_autocorr(acorr):
    """
    Finds the minimum of the auto-correlation function given by acorr
    """

    if acorr is None:
        return None

    minimum = None
    found_minimum = argrelextrema(acorr, np.less)[0]
    if len(found_minimum) == 1:
        if found_minimum[0] != 1:
            minimum = found_minimum[0]
    elif len(found_minimum) > 1:
        if argrelextrema(acorr, np.less)[0][0] != 1:
            minimum = argrelextrema(acorr, np.less)[0][0]
        else:
            minimum = argrelextrema(acorr, np.less)[0][1]

    return minimum


def compute_autocorr_struct(interspike_intervals, no_bins):
    """
    Helper method for computing all relevant auto-correlation metrics for inter-spike intervals given
    """

    autocorr_sst = None

    sorted_isi = np.sort(interspike_intervals)
    bin_dt = (sorted_isi[-1] - sorted_isi[0]) / no_bins if len(sorted_isi) > 0 else 0.01  # compute bin dt based on histogram's number of bins
    binned_isi = bin(sorted_isi, bin_dt)
    xaxis, acorr = compute_autocorr(binned_isi)
    if xaxis is not None and acorr is not None:
        autocorr_sst = {}
        minimum_sst = find_minimum_autocorr(acorr)
        autocorr_sst["xaxis"] = xaxis * bin_dt
        autocorr_sst["acorr"] = acorr
        autocorr_sst["minimum"] = minimum_sst

    return autocorr_sst


def compute_equilibrium_for_neuron_type(spike_mon, time_frame=0.05, tol=0.001):
    """
    Computes equilibrium time for a group of neurons monitored by a spike monitor.
    It does that by using a sliding window algorithm of `time_frame` and
    checking if firing rate of neurons has changed between 2 consecutive time windows
    """

    t = 0
    last_spike_t = spike_mon.t[-1] / second if len(spike_mon.t) > 0 else t

    current_firing_rate = None
    while t < last_spike_t:
        firing_rate = compute_firing_rate_for_neuron_type(spike_mon, t, t + time_frame)

        if current_firing_rate is not None and (np.mean(firing_rate) - np.mean(current_firing_rate) < tol):
            current_firing_rate = firing_rate
            break
        else:
            current_firing_rate = firing_rate
            t += time_frame

    return t, current_firing_rate


def compute_burst_mask(spikes, maxISI):
    """
    Computes a burst mask for the spikes of a given neuron.
    A spike is said to be in a burst if the inter-spike interval time is less than maxISI
    Returns a binary vector. 1 means spike is in a burst. 0 means is not
    """

    isi = np.diff(spikes)
    mask = np.zeros(len(spikes))

    for i, isi_value in enumerate(isi):
        if isi_value <= maxISI:
            # also the next index because the isi vector has offset of 1 compare to mask/spikes
            mask[i] = mask[i + 1] = 1

    return mask


def compute_burst_trains(spike_mon, maxISI):
    """
    Computes a burst mask for each neuron monitored by the spike_mon
    """

    burst_trains = {}
    for neuron_index in spike_mon.spike_trains():
        burst_mask = compute_burst_mask(spike_mon.spike_trains()[neuron_index], maxISI)
        burst_trains[neuron_index] = burst_mask

    return burst_trains


def compute_burst_lengths(burst_mask):
    """
    Given a burst mask which is iterated, a burst is defined as series of consecutive occurrences of 1.
    Each of these series is considered a burst and has an associated burst length.
    Returns a vector of bursts lengths.
    """

    burst_lengths = []
    idx = 0
    len_burst_mask = len(burst_mask)

    while idx < len_burst_mask:
        if burst_mask[idx] == 1:
            burst_length = 0

            while idx < len_burst_mask and burst_mask[idx] == 1:  # iterate till the end of series
                burst_length += 1
                idx += 1

            burst_lengths.append(burst_length)

        idx += 1

    return burst_lengths


def compute_burst_lengths_by_neuron_group(burst_trains):
    burst_lengths = []
    for neuron_index in burst_trains:
        burst_lengths_by_neuron = compute_burst_lengths(burst_trains[neuron_index])
        burst_lengths.extend(burst_lengths_by_neuron)

    return burst_lengths


def save_agg_results_to_folder(results, output_folder=None, file_name='results.json'):
    """
    Helper method for saving aggregated results for multiple simulations with different degree inputs to folder
    """

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        json_file = open(f'{output_folder}/{file_name}', 'w')
        json_file.write(json.dumps(results, indent=4))
        json_file.close()


def save_results_to_folder(results, output_folder=None, file_name='results.json'):
    """
    Helper method for saving individual simulation results to folder
    """

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dump = {}
        dump['avg_firing_rate_cs'] = np.around(np.mean(results["firing_rates_cs"]), decimals=3)
        dump['avg_firing_rate_cc'] = np.around(np.mean(results["firing_rates_cc"]), decimals=3)
        dump['avg_firing_rate_sst'] = np.around(np.mean(results["firing_rates_sst"]), decimals=3)
        dump['avg_firing_rate_pv'] = np.around(np.mean(results["firing_rates_pv"]), decimals=3)

        json_file = open(f'{output_folder}/{file_name}', 'w')
        json_file.write(json.dumps(dump, indent=4))
        json_file.close()


def distributionInput(a_data, b_data, spatialF, orientation, spatialPhase, amplitude, T, steady_input, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST.
    Using the function from textbook theoretical neurosciences.
    Turning the image into stimulus by converting the difference between that pixel over time and the
    difference between the pixel and the overall backaground level of luminance.
    Output: Stimulus to the L5 neuron
    Steady_input: making a check to make sure the input is steady.
    """

    i = 0
    inputs_p_all = []
    N_indices = [[0, N[0]], [sum(N[:1]), sum(N[:2])], [sum(N[:2]), sum(N[:3])], [sum(N[:3]), sum(N)]]
    for popu in N_indices:
        inputs_p = []

        if steady_input[i] > 0.5:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase) + amplitude[i])
            inputs_p = np.array(inputs_p)
        else:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase * t) + amplitude[i])
            inputs_p = np.array(inputs_p)
        i += 1
        inputs_p_all.append(inputs_p)

    inputs = np.concatenate((inputs_p_all), axis=1)

    return inputs


def calculate_pairwise_selectivity(input_1, input_2):
    return np.abs(input_1 - input_2) / (input_1 + input_2)


def calculate_selectivity(fire_rates):
    """
    Calculate mean and std of selectivity.
    fire rates should contain a vector of size 4, containing fire rate measurements
    for stimulus of 4 directions [0, 90, 180, 270] degrees
    """
    assert len(fire_rates) == 4

    preferred_orientation_idx = np.argmax(fire_rates)  # get the index of the maximum firing rate

    # fire rate of preferred stimulus
    fire_rate_preferred = fire_rates[preferred_orientation_idx]

    # average fire rate of preferred stimulus in both directions
    fire_rate_preferred_orientation = np.mean([
        fire_rates[preferred_orientation_idx],
        fire_rates[(preferred_orientation_idx + 2) % 4]
    ])

    # fire rate of orthogonal stimulus in both directions
    fire_rate_orthogonal_orientation = np.mean([
        fire_rates[(preferred_orientation_idx + 1) % 4],
        fire_rates[(preferred_orientation_idx + 3) % 4]
    ])

    # fire rate of opposite stimulus
    fire_rate_opposite = fire_rates[(preferred_orientation_idx + 2) % 4]

    orientation_selectivity = calculate_pairwise_selectivity(fire_rate_preferred_orientation, fire_rate_orthogonal_orientation)
    orientation_selectivity_paper = calculate_pairwise_selectivity(fire_rate_preferred, fire_rate_orthogonal_orientation)
    direction_selectivity = calculate_pairwise_selectivity(fire_rate_preferred, fire_rate_opposite)

    selectivity = {}
    selectivity["orientation"] = np.around(orientation_selectivity, decimals=3)
    selectivity["orientation_paper"] = np.around(orientation_selectivity_paper, decimals=3)
    selectivity["direction"] = np.around(direction_selectivity, decimals=3)

    return selectivity


def calculate_aggregate_results(individual_results):
    agg_results = {}
    agg_results["input_selectivity"] = 0.0  # TODO Calculate input selectivity

    fire_rates_CS = [np.mean(result["firing_rates_cs"]) for result in individual_results]
    selectivity_CS = calculate_selectivity(fire_rates_CS)
    agg_results["output_selectivity_cs"] = selectivity_CS
    agg_results['mean_fire_rate_cs'] = np.mean(fire_rates_CS)

    fire_rates_CC = [np.mean(result["firing_rates_cc"]) for result in individual_results]
    selectivity_CC = calculate_selectivity(fire_rates_CC)
    agg_results["output_selectivity_cc"] = selectivity_CC
    agg_results['mean_fire_rate_cc'] = np.mean(fire_rates_CC)

    fire_rates_PV = [np.mean(result["firing_rates_pv"]) for result in individual_results]
    selectivity_PV = calculate_selectivity(fire_rates_PV)
    agg_results["output_selectivity_pv"] = selectivity_PV
    agg_results['mean_fire_rate_pv'] = np.mean(fire_rates_PV)

    fire_rates_SST = [np.mean(result["firing_rates_sst"]) for result in individual_results]
    selectivity_SST = calculate_selectivity(fire_rates_SST)
    agg_results["output_selectivity_sst"] = selectivity_SST
    agg_results['mean_fire_rate_sst'] = np.mean(fire_rates_SST)

    if selectivity_CC["orientation"] > 0.00001 and selectivity_CC["direction"] > 0.00001:
        agg_results["os_rel"] = (selectivity_CS["orientation"] - selectivity_CC["orientation"]) / (selectivity_CS["orientation"] + selectivity_CC["orientation"])
        agg_results["ds_rel"] = (selectivity_CS["direction"] - selectivity_CC["direction"]) / (selectivity_CS["direction"] + selectivity_CC["direction"])
        agg_results["os_paper_rel"] = (selectivity_CS["orientation_paper"] - selectivity_CC["orientation_paper"]) / (selectivity_CS["orientation_paper"] + selectivity_CC["orientation_paper"])

    return agg_results
