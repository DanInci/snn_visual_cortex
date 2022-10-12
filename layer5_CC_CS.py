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

    ################################################################################

    ################################################################################
    # Define neurons & connections
    ################################################################################

    print("Defining neurons ... ")

    # SST Neurons
    sst_neurons = NeuronGroup(N_sst, model=eqs_sst_inh, threshold='v > V_t',
                              reset='v = E_l', refractory=8.3 * ms, method='euler')
    sst_neurons.set_states({'I_external': I_ext_sst})
    sst_neurons.v = 'E_l + rand()*(V_t-E_l)'
    sst_neurons.g_e = 'rand()*w_e'
    sst_neurons.g_i = 'rand()*w_i'

    # PV Neurons
    pv_neurons = NeuronGroup(N_pv, model=eqs_pv_inh, threshold='v > V_t',
                             reset='v = E_l', refractory=8.3 * ms, method='euler')
    pv_neurons.set_states({'I_external': I_ext_pv})
    pv_neurons.v = 'E_l + rand()*(V_t-E_l)'
    pv_neurons.g_e = 'rand()*w_e'
    pv_neurons.g_i = 'rand()*w_i'

    # CS Neurons
    cs_neurons = NeuronGroup(N_cs, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cs_neurons.set_states({'I_external': I_ext_cs})
    cs_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cs_neurons.v_d = -70 * mV
    cs_neurons.g_es = cs_neurons.g_ed = 'rand()*w_e'
    cs_neurons.g_is = cs_neurons.g_id = 'rand()*w_i'

    # CC Neurons
    cc_neurons = NeuronGroup(N_cc, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cc_neurons.set_states({'I_external': I_ext_cc})
    cc_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cc_neurons.v_d = -70 * mV
    cc_neurons.g_es = cc_neurons.g_ed = 'rand()*w_e'
    cc_neurons.g_is = cc_neurons.g_id = 'rand()*w_i'

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

    print("Plotting results ... ")

    plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, output_folder='output', file_name='spike_raster_plot')

    plot_states(state_mon_cs, spike_mon_cs, spike_thld=V_t, output_folder='output', file_name='state_plot_CS')
    plot_states(state_mon_cc, spike_mon_cc, spike_thld=V_t, output_folder='output', file_name='state_plot_CC')
    plot_states(state_mon_sst, spike_mon_sst, spike_thld=V_t, output_folder='output', file_name='state_plot_SST')
    plot_states(state_mon_pv, spike_mon_pv, spike_thld=V_t, output_folder='output', file_name='state_plot_PV')

    results = {}

    # Compute firing rate for each neuron group

    results["firing_rates_cs"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cs, duration)
    results["firing_rates_cc"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cc, duration)
    results["firing_rates_sst"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_sst, duration)
    results["firing_rates_pv"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_pv, duration)

    # Compute input & output selectivity for CC & CS neuron groups

    results["input_selectivity"] = hlp.compute_input_selectivity(I_ext_cs)
    results["output_selectivity_cs"] = hlp.compute_output_selectivity_for_neuron_type(spike_mon_cs, duration)
    results["output_selectivity_cc"] = hlp.compute_output_selectivity_for_neuron_type(spike_mon_cc, duration)

    # Compute inter-spike intervals for each neuron group

    results["interspike_intervals_cs"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cs), axis=0)
    results["interspike_intervals_cc"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cc), axis=0)
    results["interspike_intervals_sst"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_sst), axis=0)
    results["interspike_intervals_pv"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_pv), axis=0)

    plot_isi_histograms(results["interspike_intervals_cs"], results["interspike_intervals_cc"], results["interspike_intervals_sst"], results["interspike_intervals_pv"], output_folder='output', file_name='isi_histograms')

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
