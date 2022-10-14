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


def count_spikes_for_neuron_type(spike_mon):
    a = pd.Series([i for i in spike_mon.i], dtype=int)
    return a.value_counts()


@check_units(sim_duration=ms)
def compute_firing_rate_for_neuron_type(spike_mon, sim_duration):
    spikes_for_i = count_spikes_for_neuron_type(spike_mon)
    return (spikes_for_i / sim_duration) * Hz


def compute_input_selectivity(inputs):
    assert len(inputs) >= 2

    return np.abs(inputs[0] - inputs[1]) / (inputs[0] + inputs[1])


@check_units(sim_duration=ms, result=1)
def compute_output_selectivity_for_neuron_type(spike_mon, sim_duration):
    rates_for_i = compute_firing_rate_for_neuron_type(spike_mon, sim_duration)
    assert len(rates_for_i) >= 2

    return np.abs(rates_for_i[0] - rates_for_i[1]) / (rates_for_i[0] + rates_for_i[1])


def compute_interspike_intervals(spike_mon):
    by_neuron = []

    for neuron_index in spike_mon.spike_trains():
        interspike_intervals = np.diff(spike_mon.spike_trains()[neuron_index])
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


def compute_autocorr_struct(interspike_intervals, bin_size=0.005):
    autocorr_sst = None

    binned_isi = bin(np.sort(interspike_intervals), bin_size)
    xaxis, acorr = compute_autocorr(binned_isi)
    if xaxis is not None and acorr is not None:
        autocorr_sst = {}
        minimum_sst = find_minimum_autocorr(acorr)
        autocorr_sst["xaxis"] = xaxis * bin_size
        autocorr_sst["acorr"] = acorr
        autocorr_sst["minimum"] = minimum_sst

    return autocorr_sst
