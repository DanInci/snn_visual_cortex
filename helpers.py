from brian2 import *
import pandas as pd


# TODO see how to reference this from equations
@check_units(x=volt, result=1)
def sigmoid(x):
    ### Sigmoid function params
    E_d = -38 * mV  # position control of threshold
    D_d = 6 * mV  # sharpness control of threshold

    return 1/(1+exp(-(-x-E_d)/D_d))


def count_spikes_for_neuron_type(spike_mon):
    a = pd.Series([i for i in spike_mon.i], dtype=int)
    return a.value_counts()


@check_units(sim_duration=ms)
def compute_firing_rate_for_neuron_type(spike_mon, sim_duration):
    spikes_for_i = count_spikes_for_neuron_type(spike_mon)
    return (spikes_for_i / sim_duration) * Hz


def compute_input_selectivity(inputs):
    return np.abs(inputs[0] - inputs[1]) / (inputs[0] + inputs[1])


@check_units(sim_duration=ms, result=1)
def compute_output_selectivity_for_neuron_type(spike_mon, sim_duration):
    rates_for_i = compute_firing_rate_for_neuron_type(spike_mon, sim_duration)
    return np.abs(rates_for_i[0] - rates_for_i[1]) / (rates_for_i[0] + rates_for_i[1])


def compute_interspike_intervals(spike_mon):
    by_neuron = []

    for neuron_index in spike_mon.spike_trains():
        interspike_intervals = np.diff(spike_mon.spike_trains()[neuron_index])
        by_neuron.append(interspike_intervals)

    return by_neuron
