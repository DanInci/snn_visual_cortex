from brian2 import *
import pandas as pd

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
    spike_counts = {}

    for index in spike_mon.spike_trains():
        a = pd.Series(spike_mon.spike_trains()[index] / second)

        if from_t is not None and to_t is not None:
            a = a.loc[lambda x: (x >= from_t) & (x <= to_t)]

        spike_counts[index] = a.size

    return pd.Series(spike_counts)


@check_units(sim_duration=ms)
def compute_firing_rate_for_neuron_type(spike_mon, from_t, to_t):
    spikes_for_i = count_spikes_for_neuron_type(spike_mon, from_t, to_t)
    duration = to_t - from_t

    return spikes_for_i / duration


def compute_input_selectivity(inputs):
    assert len(inputs) >= 2

    return np.abs(inputs[0] - inputs[1]) / (inputs[0] + inputs[1])


@check_units(sim_duration=ms, result=1)
def compute_output_selectivity_for_neuron_type(spike_mon, from_t, to_t):
    rates_for_i = compute_firing_rate_for_neuron_type(spike_mon, from_t, to_t)
    assert len(rates_for_i) >= 2

    return np.abs(rates_for_i[0] - rates_for_i[1]) / (rates_for_i[0] + rates_for_i[1])


def compute_interspike_intervals(spike_mon, from_t, to_t):
    by_neuron = []

    for neuron_index in spike_mon.spike_trains():
        spikes_for_neuron = pd.Series(spike_mon.spike_trains()[neuron_index] / second)
        filterd_by_period = spikes_for_neuron.loc[lambda x: (x >= from_t) & (x <= to_t)]

        interspike_intervals = np.diff(filterd_by_period)
        by_neuron.append(interspike_intervals)

    return by_neuron


def compute_autocorr(spike_intervals):
    if len(spike_intervals) == 0:
        return None, None

    autocorr = plt.acorr(spike_intervals, normed=True, maxlags=None)
    right_xaxis = autocorr[0][int(len(autocorr[1]) / 2):]
    right_acorr = autocorr[1][int(len(autocorr[1]) / 2):]
    return right_xaxis, right_acorr


def find_minimum_autocorr(acorr):
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


def compute_equilibrium_for_neuron_type(spike_mon, time_frame):
    t = 0
    last_spike_t = spike_mon.t[-1] / second if len(spike_mon.t) > 0 else t

    current_firing_rate = None
    while t < last_spike_t:
        firing_rate = compute_firing_rate_for_neuron_type(spike_mon, t, t + time_frame)

        if current_firing_rate is not None and firing_rate.equals(current_firing_rate) and current_firing_rate[
            current_firing_rate == 0].size == 0:
            current_firing_rate = firing_rate
            break
        else:
            current_firing_rate = firing_rate
            t += time_frame

    return t, current_firing_rate


def compute_burst_mask(spikes, maxISI):
    isi = np.diff(spikes)
    mask = np.zeros(len(spikes))

    for i, isi_value in enumerate(isi):
        if isi_value <= maxISI:
            # also the next index because the isi vector has offset of 1 compare to mask/spikes
            mask[i] = mask[i + 1] = 1

    return mask


def compute_burst_trains(spike_mon, maxISI):
    burst_trains = {}
    for neuron_index in spike_mon.spike_trains():
        burst_mask = compute_burst_mask(spike_mon.spike_trains()[neuron_index], maxISI)
        burst_trains[neuron_index] = burst_mask

    return burst_trains


def compute_burst_lengths(burst_mask):
    "count how many series of consecutive occurances of 1 appear"

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
