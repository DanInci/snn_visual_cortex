from brian2 import *
from plotting import *
from equations import *
from parameters import default as default_params

import helpers as hlp


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run_simulation(params=None, seed_val=12345, sst_target_soma=True):
    p = Struct(**params)

    start_scope()
    seed(seed_val)

    ################################################################################
    # Model parameters
    ################################################################################
    ### General parameters
    time_frame = 0.2  # Time frame for computing equlibrium time
    no_bins = 10  # Number of bins for interspike intervals historgram
    plot_only_from_equilibrium = True  # Plot graphs only from equilibrium time

    duration = p.duration  # Total simulation time
    sim_dt = p.sim_dt  # Integrator/sampling step

    N_sst = p.N_sst  # Number of SST neurons (inhibitory)
    N_pv = p.N_pv  # Number of PV neurons (inhibitory)
    N_cc = p.N_cc  # Number of CC neurons (excitatory)
    N_cs = p.N_cs  # Number of CS neurons (excitatory)

    ### Neuron parameters
    tau_S = p.tau_S  #
    tau_D = p.tau_D  #
    tau_SST = p.tau_SST  #
    tau_PV = p.tau_PV  #
    tau_E = p.tau_E  # Excitatory synaptic time constant
    tau_I = p.tau_I  # Inhibitory synaptic time constant

    C_S = p.C_S
    C_D = p.C_D
    C_SST = p.C_SST
    C_PV = p.C_PV

    E_l = p.E_l  # leak reversal potential
    E_e = p.E_e  # Excitatory synaptic reversal potential
    E_i = p.E_i  # Inhibitory synaptic reversal potential

    V_t = p.V_t  # spiking threashold
    V_r = p.V_r  # reset potential

    c_d = p.c_d  # back-propagates somatic spikes to to the dendrites
    g_s = p.g_s  # propagates dendritic regenerative activity to soma
    g_d = p.g_d  # propagates dendritic regenerative activity to denderites

    ### Sigmoid function params
    E_d = p.E_d  # position control of threshold
    D_d = p.D_d  # sharpness control of threshold

    ### Synapse parameters
    w_e = p.w_e  # Excitatory synaptic conductance
    w_i = p.w_i  # Inhibitory synaptic conductance

    ### External Input
    I_ext_sst = p.I_ext_sst
    I_ext_pv = p.I_ext_pv
    I_ext_cs = p.I_ext_cs
    I_ext_cc = p.I_ext_cc

    lambda_cc = p.lambda_cc
    lambda_cs = p.lambda_cs

    ################################################################################

    ################################################################################
    # Define neurons & connections
    ################################################################################

    print("Defining neurons ... ")

    # SST Neurons
    sst_neurons = NeuronGroup(N_sst, model=eqs_sst_inh, threshold='v > V_t',
                              reset='v = E_l', refractory=8.3 * ms, method='euler')
    sst_neurons.v = 'E_l + rand()*(V_t-E_l)'
    sst_neurons.g_e = 'rand()*w_e'
    sst_neurons.g_i = 'rand()*w_i'

    # PV Neurons
    pv_neurons = NeuronGroup(N_pv, model=eqs_pv_inh, threshold='v > V_t',
                             reset='v = E_l', refractory=8.3 * ms, method='euler')
    pv_neurons.v = 'E_l + rand()*(V_t-E_l)'
    pv_neurons.g_e = 'rand()*w_e'
    pv_neurons.g_i = 'rand()*w_i'

    # CS Neurons
    cs_neurons = NeuronGroup(N_cs, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cs_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cs_neurons.v_d = -70 * mV
    cs_neurons.g_es = cs_neurons.g_ed = 'rand()*w_e'
    cs_neurons.g_is = cs_neurons.g_id = 'rand()*w_i'

    # Poisson input to CS neurons
    cs_neurons_p1 = PoissonInput(cs_neurons[0], 'g_es', N=1, rate=lambda_cs, weight=I_ext_cs[0])
    cs_neurons_p2 = PoissonInput(cs_neurons[1], 'g_es', N=1, rate=lambda_cs, weight=I_ext_cs[1])

    # CC Neurons
    cc_neurons = NeuronGroup(N_cc, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cc_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cc_neurons.v_d = -70 * mV
    cc_neurons.g_es = cc_neurons.g_ed = 'rand()*w_e'
    cc_neurons.g_is = cc_neurons.g_id = 'rand()*w_i'

    # Poisson input to CC neurons
    cc_neurons_p1 = PoissonInput(cc_neurons[0], 'g_es', N=1, rate=lambda_cc, weight=I_ext_cc[0])
    cc_neurons_p2 = PoissonInput(cc_neurons[1], 'g_es', N=1, rate=lambda_cc, weight=I_ext_cc[1])

    # ##############################################################################
    # # Synapses & Connections
    # ##############################################################################

    print("Defining synapses ... ")

    # SST <=> PV
    conn_SST_PV = Synapses(sst_neurons, pv_neurons, on_pre='g_i+=w_i', name='SST_PV')  # inhibitory
    conn_SST_PV.connect(p=p.pSST_PV)
    conn_PV_SST = Synapses(pv_neurons, sst_neurons, on_pre='g_i+=w_i', name='PV_SST')  # inhibitory
    conn_PV_SST.connect(p=p.pPV_SST)

    # PV <=> PYR soma
    ## target CS soma
    conn_PV_CSsoma = Synapses(pv_neurons, cs_neurons, on_pre='g_is+=w_i', name='PV_CSsoma')  # inhibitory
    conn_PV_CSsoma.connect(p=p.pPV_CS)
    conn_CSsoma_PV = Synapses(cs_neurons, pv_neurons, on_pre='g_e+=w_e', name='CSsoma_PV')  # excitatory
    conn_CSsoma_PV.connect(p=p.pCS_PV)

    ## target CC soma
    conn_PV_CCsoma = Synapses(pv_neurons, cc_neurons, on_pre='g_is+=w_i', name='PV_CCsoma')  # inhibitory
    conn_PV_CCsoma.connect(p=p.pPV_CC)
    conn_CCsoma_PV = Synapses(cc_neurons, pv_neurons, on_pre='g_e+=w_e', name='CCsoma_PV')  # excitatory
    conn_CCsoma_PV.connect(p=p.pCC_PV)

    # SST <=> PYR soma
    ## target CS soma
    conn_SST_CSsoma = Synapses(sst_neurons, cs_neurons, on_pre='g_is+=w_i',
                               name='SST_CSsoma')  # inhibitory (optional connection)
    conn_SST_CSsoma.connect(p=p.pSST_CS if sst_target_soma else 0) # inhibitory (optional connection)
    conn_CSsoma_SST = Synapses(cs_neurons, sst_neurons, on_pre='g_e+=w_e', name='CSsoma_SST')  # excitatory
    conn_CSsoma_SST.connect(p=p.pCS_SST)

    ## taget CC soma
    conn_SST_CCsoma = Synapses(sst_neurons, cc_neurons, on_pre='g_is+=w_i',
                               name='SST_CCsoma')  # inhibitory (optional connection)
    conn_SST_CCsoma.connect(p=p.pSST_CC if sst_target_soma else 0)  # inhibitory (optional connection)
    conn_CCsoma_SST = Synapses(cc_neurons, sst_neurons, on_pre='g_e+=w_e', name='CCsoma_SST')  # excitatory
    conn_CCsoma_SST.connect(p=p.pCC_SST)

    # CC => CS
    ## target CS soma
    conn_CC_CS = Synapses(cc_neurons, cs_neurons, on_pre='g_es+=w_e', name='CC_CSsoma')  # excitatory
    conn_CC_CS.connect(p=p.pCC_CS)

    # self connections
    conn_CSsoma_CSsoma = Synapses(cs_neurons, cs_neurons, on_pre='g_es+=w_e', name='CSsoma_CSsoma')  # excitatory
    conn_CSsoma_CSsoma.connect(p=p.pCS_CS)
    backprop_CS = Synapses(cs_neurons, cs_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CS')
    backprop_CS.connect(condition='i==j')  # Connect all CS neurons to themselves

    conn_CCsoma_CCsoma = Synapses(cc_neurons, cc_neurons, on_pre='g_es+=w_e', name='CCsoma_CCsoma')  # excitatory
    conn_CCsoma_CCsoma.connect(p=p.pCC_CC)
    backprop_CC = Synapses(cc_neurons, cc_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CC')
    backprop_CC.connect(condition='i==j')  # Connect all CC neurons to themselves

    conn_SST_SST = Synapses(sst_neurons, sst_neurons, on_pre='g_i+=w_i', name='SST_SST')  # inhibitory
    conn_SST_SST.connect(p=p.pSST_SST)

    conn_PV_PV = Synapses(pv_neurons, pv_neurons, on_pre='g_i+=w_i', name='PV_PV')  # inhibitory
    conn_PV_PV.connect(p=p.pPV_PV)

    # SST => PYR dendrite
    ## target CS dendrite
    conn_SST_CSdendrite = Synapses(sst_neurons, cs_neurons, on_pre='g_id+=w_i', name='SST_CSdendrite')  # inhibitory
    conn_SST_CSdendrite.connect(p=p.pSST_CS)  # not sure about this here

    ## target CC dendrite
    conn_SST_CCdendrite = Synapses(sst_neurons, cc_neurons, on_pre='g_id+=w_i', name='SST_CCdendrite')  # inhibitory
    conn_SST_CCdendrite.connect(p=p.pSST_CC)  # not sure about this here

    # ##############################################################################
    # # Monitors
    # ##############################################################################

    print("Defining monitors ... ")

    # Record spikes of different neuron groups
    spike_mon_sst = SpikeMonitor(sst_neurons)
    spike_mon_pv = SpikeMonitor(pv_neurons)
    spike_mon_cs = SpikeMonitor(cs_neurons)
    spike_mon_cc = SpikeMonitor(cc_neurons)

    # Record conductances and membrane potential of neuron ni
    state_mon_sst = StateMonitor(sst_neurons, ['v', 'g_e', 'g_i'], record=[0])
    state_mon_pv = StateMonitor(pv_neurons, ['v', 'g_e', 'g_i'], record=[0])
    state_mon_cs = StateMonitor(cs_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])
    state_mon_cc = StateMonitor(cc_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])

    # ##############################################################################
    # # Simulation run
    # ##############################################################################

    defaultclock.dt = sim_dt

    run(duration, report='text')

    ################################################################################
    # Analysis and plotting
    ################################################################################

    equilibrium_times = []
    for idx, spike_mon in enumerate([spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv]):
        t, firing_rate = hlp.compute_equilibrium_for_neuron_type(spike_mon, time_frame)
        if firing_rate is not None:
            print(f"Found for {index_to_ntype_dict[idx]} neurons")

        equilibrium_times.append(t)

    equilibrium_t = max(equilibrium_times) * second
    if equilibrium_t < duration:
        print(f"Equilibrium for all neurons start at: {equilibrium_t}")
    else:
        print(f"WARNING: Equilibrium was not found during the duration of the simulation")

    # Only compute properties of the system from equilibrium time to end simulation time
    from_t = equilibrium_t
    to_t = duration

    print("Plotting results from equilibrium ... ")

    plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, plot_only_from_equilibrium, from_t, to_t, output_folder='output', file_name='spike_raster_plot')

    plot_states(state_mon_cs, spike_mon_cs, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder='output', file_name='state_plot_CS')
    plot_states(state_mon_cc, spike_mon_cc, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder='output', file_name='state_plot_CC')
    plot_states(state_mon_sst, spike_mon_sst, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder='output', file_name='state_plot_SST')
    plot_states(state_mon_pv, spike_mon_pv, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder='output', file_name='state_plot_PV')

    results = {}

    # Compute firing rate for each neuron group

    results["firing_rates_cs"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cs, from_t, to_t)
    results["firing_rates_cc"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cc, from_t, to_t)
    results["firing_rates_sst"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_sst, from_t, to_t)
    results["firing_rates_pv"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_pv, from_t, to_t)

    # Compute input & output selectivity for CC & CS neuron groups

    results["input_selectivity"] = hlp.compute_input_selectivity(I_ext_cs)
    results["output_selectivity_cs"] = hlp.compute_output_selectivity_for_neuron_type(spike_mon_cs, from_t, to_t)
    results["output_selectivity_cc"] = hlp.compute_output_selectivity_for_neuron_type(spike_mon_cc, from_t, to_t)

    # Compute inter-spike intervals for each neuron group

    results["interspike_intervals_cs"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cs, from_t, to_t), axis=0)
    results["interspike_intervals_cc"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cc, from_t, to_t), axis=0)
    results["interspike_intervals_sst"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_sst, from_t, to_t), axis=0)
    results["interspike_intervals_pv"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_pv, from_t, to_t), axis=0)

    # Compute auto-correlation for isi for each neuron group
    # for CS
    autocorr_cs = hlp.compute_autocorr_struct(results["interspike_intervals_cs"], no_bins)
    if autocorr_cs:
        results["acorr_min_cs"] = autocorr_cs["minimum"]

    # for CC
    autocorr_cc = hlp.compute_autocorr_struct(results["interspike_intervals_cc"], no_bins)
    if autocorr_cc:
        results["acorr_min_cc"] = autocorr_cc["minimum"]


    # for SST
    autocorr_sst = hlp.compute_autocorr_struct(results["interspike_intervals_sst"], no_bins)
    if autocorr_sst:
        results["acorr_min_sst"] = autocorr_sst["minimum"]

    # for PV
    autocorr_pv = hlp.compute_autocorr_struct(results["interspike_intervals_pv"], no_bins)
    if autocorr_pv:
        results["acorr_min_sst"] = autocorr_pv["minimum"]

    interspike_intervals = [results["interspike_intervals_cs"], results["interspike_intervals_cc"], results["interspike_intervals_sst"], results["interspike_intervals_pv"]]
    autocorr = [autocorr_cs, autocorr_cc, autocorr_sst, autocorr_pv]
    plot_isi_histograms(interspike_intervals, no_bins, autocorr=autocorr, output_folder='output', file_name='isi_histograms')

    # Detect bursts
    # for CS
    if autocorr_cs:
        maxISI_cs = autocorr_cs["xaxis"][autocorr_cs["minimum"]] if autocorr_cs["minimum"] else None
        burst_trains_cs = hlp.compute_burst_trains(spike_mon_cs, maxISI_cs * second) if maxISI_cs else {}
        results["burst_lengths_cs"] = hlp.compute_burst_lengths_by_neuron_group(burst_trains_cs)

    # for CC
    if autocorr_cc:
        maxISI_cc = autocorr_cc["xaxis"][autocorr_cc["minimum"]] if autocorr_cc["minimum"] else None
        burst_trains_cc = hlp.compute_burst_trains(spike_mon_cc, maxISI_cc * second) if maxISI_cc else {}
        results["burst_lengths_cc"] = hlp.compute_burst_lengths_by_neuron_group(burst_trains_cc)

    # for SST
    if autocorr_sst:
        maxISI_sst = autocorr_sst["xaxis"][autocorr_sst["minimum"]] if autocorr_sst["minimum"] else None
        burst_trains_sst = hlp.compute_burst_trains(spike_mon_sst, maxISI_sst * second) if maxISI_sst else {}
        results["burst_lengths_sst"] = hlp.compute_burst_lengths_by_neuron_group(burst_trains_sst)

    # for PV
    if autocorr_pv:
        maxISI_pv = autocorr_pv["xaxis"][autocorr_pv["minimum"]] if autocorr_pv["minimum"] else None
        burst_trains_pv = hlp.compute_burst_trains(spike_mon_pv, maxISI_pv * second) if maxISI_pv else {}
        results["burst_lengths_pv"] = hlp.compute_burst_lengths_by_neuron_group(burst_trains_pv)

    return results


params = default_params
results = run_simulation(params, seed_val=12345)

print(f'Avg firing rate for CS neurons: {np.mean(results["firing_rates_cs"]) * Hz}')
print(f'Avg firing rate for CC neurons: {np.mean(results["firing_rates_cc"]) * Hz}')
print(f'Avg firing rate for SST neurons: {np.mean(results["firing_rates_sst"]) * Hz}')
print(f'Avg firing rate for PV neurons: {np.mean(results["firing_rates_pv"]) * Hz}')

print(f'Input selectivity: {results["input_selectivity"]}')
print(f'Output selectivity CS: {results["output_selectivity_cs"]}')
print(f'Output selectivity CC: {results["output_selectivity_cc"]}')

print(f'Burst lengths vector for CS neurons: {results.get("burst_lengths_cs")}')
print(f'Burst lengths vector for CC neurons: {results.get("burst_lengths_cc")}')
print(f'Burst lengths vector for SST neurons: {results.get("burst_lengths_sst")}')
print(f'Burst lengths vector for PV neurons: {results.get("burst_lengths_pv")}')
