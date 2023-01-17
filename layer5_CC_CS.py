from brian2 import *
from plotting import *
from equations import *
import copy

import helpers as hlp


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def analyse_network_simulation(spike_monitors, state_monitors, synapses, p, output_folder=None):
    """
    Does an analysis of a simulation post-run. Saves plots to `output_folder`
    Returns the results of simulation analysis
    """

    spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc = spike_monitors
    state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc = state_monitors

    ################################################################################
    # Compute equilibrium time of simulation
    ################################################################################

    if p.recompute_equilibrium:
        equilibrium_times = []
        for idx, spike_mon in enumerate([spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv]):
            t, firing_rate = hlp.compute_equilibrium_for_neuron_type(spike_mon)
            if firing_rate is not None:
                print(f"- Equilibrium found for {index_to_ntype_dict[idx]} neurons")

            equilibrium_times.append(t)

        equilibrium_t = max(equilibrium_times) * second
        if equilibrium_t < p.duration:
            print(f"* Equilibrium for all neurons start at: {equilibrium_t}")
        else:
            print(f"WARNING: Equilibrium was not found during the duration of the simulation")
    else:
        print(f"* Skipping recalculating equilibrium time. Using default equilibrium time={p.default_equilibrium_t}")
        equilibrium_t = p.default_equilibrium_t

    # Only compute properties of the system from equilibrium time to end simulation time
    from_t = equilibrium_t
    to_t = p.duration

    ################################################################################
    # Analysis and plotting
    ################################################################################

    raster_from_t = max(max(from_t, to_t - 3 * second), 0)
    raster_to_t = to_t
    plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, raster_from_t, raster_to_t,
                output_folder=output_folder, file_name='spike_raster_plot')

    plot_states(state_mon_cs, spike_mon_cs, p.V_t, from_t, to_t,
                output_folder=output_folder, file_name='state_plot_CS')
    plot_states(state_mon_cc, spike_mon_cc, p.V_t, from_t, to_t,
                output_folder=output_folder, file_name='state_plot_CC')
    plot_states(state_mon_sst, spike_mon_sst, p.V_t, from_t, to_t,
                output_folder=output_folder, file_name='state_plot_SST')
    plot_states(state_mon_pv, spike_mon_pv, p.V_t, from_t, to_t,
                output_folder=output_folder, file_name='state_plot_PV')

    # Plot connectivity graph
    if p.plot_connectivity_graph:
        plot_neuron_connectivity(synapses, output_folder=output_folder, file_name='neuron_connectivity')

    results = {}

    # Compute firing rate for each neuron group

    results["firing_rates_cs"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cs, from_t, to_t)
    results["firing_rates_cc"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cc, from_t, to_t)
    results["firing_rates_sst"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_sst, from_t, to_t)
    results["firing_rates_pv"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_pv, from_t, to_t)

    firing_rates = [results["firing_rates_cs"], results["firing_rates_cc"], results["firing_rates_sst"],
                    results["firing_rates_pv"]]
    plot_firing_rate_histograms(firing_rates, p.no_bins_firing_rates, output_folder=output_folder,
                                file_name='firing_rate_histograms')

    # Compute inter-spike intervals for each neuron group

    results["interspike_intervals_cs"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cs, from_t, to_t),
                                                        axis=0)
    results["interspike_intervals_cc"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_cc, from_t, to_t),
                                                        axis=0)
    results["interspike_intervals_sst"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_sst, from_t, to_t),
                                                         axis=0)
    results["interspike_intervals_pv"] = np.concatenate(hlp.compute_interspike_intervals(spike_mon_pv, from_t, to_t),
                                                        axis=0)

    # Compute auto-correlation for isi for each neuron group
    # for CS
    autocorr_cs = hlp.compute_autocorr_struct(results["interspike_intervals_cs"], p.no_bins_isi)
    if autocorr_cs:
        results["acorr_min_cs"] = autocorr_cs["minimum"]

    # for CC
    autocorr_cc = hlp.compute_autocorr_struct(results["interspike_intervals_cc"], p.no_bins_isi)
    if autocorr_cc:
        results["acorr_min_cc"] = autocorr_cc["minimum"]

    # for SST
    autocorr_sst = hlp.compute_autocorr_struct(results["interspike_intervals_sst"], p.no_bins_isi)
    if autocorr_sst:
        results["acorr_min_sst"] = autocorr_sst["minimum"]

    # for PV
    autocorr_pv = hlp.compute_autocorr_struct(results["interspike_intervals_pv"], p.no_bins_isi)
    if autocorr_pv:
        results["acorr_min_sst"] = autocorr_pv["minimum"]

    interspike_intervals = [results["interspike_intervals_cs"], results["interspike_intervals_cc"],
                            results["interspike_intervals_sst"], results["interspike_intervals_pv"]]
    autocorr = [autocorr_cs, autocorr_cc, autocorr_sst, autocorr_pv]
    plot_isi_histograms(interspike_intervals, p.no_bins_isi, autocorr=autocorr, output_folder=output_folder,
                        file_name='isi_histograms')

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


def run_simulation_without_exh_dendrite(network, neurons, synapses, p, base_output_folder, use_synaptic_probabilities):
    """
    Runs simulation for network topology with PYR cells having ONLY somas and NO dendrites.
    Also analyses simulation run and returns results.
    """

    sst_neurons, pv_neurons, cs_neurons, cc_neurons = neurons

    E_l = p.E_l  # leak reversal potential
    V_t = p.V_t  # spiking threashold

    ### External Input parameters
    I_ext_sst = TimedArray(p.I_ext_sst*nS, dt=p.sim_dt)
    I_ext_pv = TimedArray(p.I_ext_pv*nS, dt=p.sim_dt)
    I_ext_cs = TimedArray(p.I_ext_cs*nS, dt=p.sim_dt)
    I_ext_cc = TimedArray(p.I_ext_cc*nS, dt=p.sim_dt)

    # ##############################################################################
    # Add extra synapses (pSST_CS/CC_soma)
    # ##############################################################################

    ## target CS soma
    conn_SST_CSsoma = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_is+=w*nS', name='SST_CSsoma')  # inhibitory (optional connection)
    conn_SST_CSsoma.connect(p=p.pSST_CS * p.pSST_CS_weight if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
    conn_SST_CSsoma.w = p.wSST_CS
    synapses["SST_CSsoma"] = conn_SST_CSsoma

    ## target CC soma
    conn_SST_CCsoma = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_is+=w*nS' ,name='SST_CCsoma')  # inhibitory (optional connection)
    conn_SST_CCsoma.connect(p=p.pSST_CC * p.pSST_CC_weight if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
    conn_SST_CCsoma.w = p.wSST_CC
    synapses["SST_CCsoma"] = conn_SST_CCsoma

    extra_connections = [conn_SST_CSsoma, conn_SST_CCsoma]

    # ##############################################################################
    # Define Monitors (pSST_CS/CC_soma)
    # ##############################################################################

    # Record spikes of different neuron groups
    spike_mon_sst = SpikeMonitor(sst_neurons)
    spike_mon_pv = SpikeMonitor(pv_neurons)
    spike_mon_cs = SpikeMonitor(cs_neurons)
    spike_mon_cc = SpikeMonitor(cc_neurons)

    spike_monitors = [spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc]

    # Record conductances and membrane potential of neuron groups
    inh_neuron_variables = ['v', 'g_e', 'g_i']
    state_mon_sst = StateMonitor(sst_neurons, inh_neuron_variables, record=[0])
    state_mon_pv = StateMonitor(pv_neurons, inh_neuron_variables, record=[0])

    exc_neuron_variables = ['v_s', 'g_es', 'g_is']
    state_mon_cs = StateMonitor(cs_neurons, exc_neuron_variables, record=[0])
    state_mon_cc = StateMonitor(cc_neurons, exc_neuron_variables, record=[0])

    state_monitors = [state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc]

    # ##############################################################################
    # Run Network (pSST_CS/CC_soma)
    # ##############################################################################

    print(f'* Run network simulation (pSST_CS/CC_soma - case 1CS / 1CC - no exh dendrites)')
    network.restore('initialized')

    # Add extras to network
    network.add(extra_connections)
    network.add(spike_monitors)
    network.add(state_monitors)

    network.run(p.duration, report='text')

    output_folder = f'{base_output_folder}/sst_soma_1CS_1CC' if base_output_folder else None
    result = analyse_network_simulation(spike_monitors, state_monitors, synapses, p, output_folder=output_folder)

    # Cleanup extras from network
    network.remove(extra_connections)
    network.remove(spike_monitors)
    network.remove(state_monitors)

    return result


def run_simulation_for_weighted_sst_soma_dendrite(network, neurons, synapses, p, p_SST_CS_soma, p_SST_CC_soma, base_output_folder, use_synaptic_probabilities):
    """
    Runs simulation for network topology with PYR cells having BOTH somas and dendrites.
    Connection probability of SST->CC/CS Soma is weighted through parameters `p_SST_CS_soma` and `p_SST_CC_soma`
    Connection probability of SST->CC/CS Dendrite is given by `1 - p_SST_CS_soma` and `1 - p_SST_CC_soma`
    Also analyses simulation run and returns results.
    """

    sst_neurons, pv_neurons, cs_neurons, cc_neurons = neurons

    E_l = p.E_l  # leak reversal potential
    V_t = p.V_t  # spiking threashold

    ### External Input parameters
    I_ext_sst = TimedArray(p.I_ext_sst*nS, dt=p.sim_dt)
    I_ext_pv = TimedArray(p.I_ext_pv*nS, dt=p.sim_dt)
    I_ext_cs = TimedArray(p.I_ext_cs*nS, dt=p.sim_dt)
    I_ext_cc = TimedArray(p.I_ext_cc*nS, dt=p.sim_dt)

    # ##############################################################################
    # Add extra synapses (pSST_CS/CC_soma)
    # ##############################################################################

    conn_SST_CSsoma = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_is+=w*nS',
                               name='SST_CSsoma')  # inhibitory (optional connection)
    conn_SST_CSsoma.connect(
        p=p.pSST_CS * p.pSST_CS_weight * p_SST_CS_soma if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
    conn_SST_CSsoma.w = p.wSST_CS
    synapses["SST_CSsoma"] = conn_SST_CSsoma

    ## target CS dendrite
    conn_SST_CSdendrite = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                   name='SST_CSdendrite')  # inhibitory
    conn_SST_CSdendrite.connect(
        p=p.pSST_CS * p.pSST_CS_weight * (1 - p_SST_CS_soma) if use_synaptic_probabilities else 1)
    conn_SST_CSdendrite.w = p.wSST_CS
    synapses["SST_CSdendrite"] = conn_SST_CSdendrite

    ## target CC soma
    conn_SST_CCsoma = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_is+=w*nS',
                               name='SST_CCsoma')  # inhibitory (optional connection)
    conn_SST_CCsoma.connect(
        p=p.pSST_CC * p.pSST_CC_weight * p_SST_CC_soma if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
    conn_SST_CCsoma.w = p.wSST_CC
    synapses["SST_CCsoma"] = conn_SST_CCsoma

    ## target CC dendrite
    conn_SST_CCdendrite = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                   name='SST_CCdendrite')  # inhibitory
    conn_SST_CCdendrite.connect(
        p=p.pSST_CC * p.pSST_CC_weight * (1 - p_SST_CC_soma) if use_synaptic_probabilities else 1)
    conn_SST_CCdendrite.w = p.wSST_CC
    synapses["SST_CCdendrite"] = conn_SST_CCdendrite

    extra_connections = [conn_SST_CSsoma, conn_SST_CSdendrite, conn_SST_CCsoma, conn_SST_CCdendrite]

    # ##############################################################################
    # Define Monitors (pSST_CS/CC_soma)
    # ##############################################################################

    # Record spikes of different neuron groups
    spike_mon_sst = SpikeMonitor(sst_neurons)
    spike_mon_pv = SpikeMonitor(pv_neurons)
    spike_mon_cs = SpikeMonitor(cs_neurons)
    spike_mon_cc = SpikeMonitor(cc_neurons)

    spike_monitors = [spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc]

    # Record conductances and membrane potential of neuron groups
    inh_neuron_variables = ['v', 'g_e', 'g_i']
    state_mon_sst = StateMonitor(sst_neurons, inh_neuron_variables, record=[0])
    state_mon_pv = StateMonitor(pv_neurons, inh_neuron_variables, record=[0])

    exc_neuron_variables = ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id']
    state_mon_cs = StateMonitor(cs_neurons, exc_neuron_variables, record=[0])
    state_mon_cc = StateMonitor(cc_neurons, exc_neuron_variables, record=[0])

    state_monitors = [state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc]

    # ##############################################################################
    # Run Network (pSST_CS/CC_soma)
    # ##############################################################################

    print(f'* Run network simulation (pSST_CS/CC_soma - case {p_SST_CS_soma}CS / {p_SST_CC_soma}CC)')
    network.restore('initialized')

    # Add extras to network
    network.add(extra_connections)
    network.add(spike_monitors)
    network.add(state_monitors)

    network.run(p.duration, report='text')

    output_folder = f'{base_output_folder}/sst_soma_{p_SST_CS_soma}CS_{p_SST_CC_soma}CC' if base_output_folder else None
    result = analyse_network_simulation(spike_monitors, state_monitors, synapses, p, output_folder=output_folder)

    # Cleanup extras from network
    network.remove(extra_connections)
    network.remove(spike_monitors)
    network.remove(state_monitors)

    return result


def run_simulation_for_input(params, use_synaptic_probabilities, use_dendrite_model, seed_val, base_output_folder=None):
    """
    Given external input and network parameters (through `params`) simulations are run accordingly.
    If `use_dendrite_model` multiple simulations are run for each (pSST_CS_soma, pSST_CC_soma) pairs
    If NOT `use_dendrite_model` a single simulation is run.
    Returns vector of results of simulations
    """

    p = Struct(**params)

    start_scope()
    seed(seed_val)

    E_l = p.E_l  # leak reversal potential
    V_t = p.V_t  # spiking threshold

    assert len(p.pSST_CS_soma) == len(p.pSST_CC_soma)  # since they are taken in pairs

    ################################################################################
    # Define neurons
    ################################################################################

    # SST Neurons
    sst_equations = Equations(eqs_sst_inh,
                              tau_SST=p.tau_SST, tau_E=p.tau_E, tau_I=p.tau_I,
                              E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                              C_SST=p.C_SST)
    sst_neurons = NeuronGroup(p.N_sst, model=sst_equations, threshold='v > V_t',
                              reset='v = E_l', refractory=8.3 * ms, method='euler')
    sst_neurons.v = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to SST neurons
    for n_idx in range(p.N_sst):
        sst_input_i = PoissonInput(sst_neurons, 'g_e', N=1, rate=p.lambda_sst, weight=f'I_ext_sst(t, {n_idx})')

    # PV Neurons
    pv_equations = Equations(eqs_pv_inh,
                             tau_PV=p.tau_PV, tau_E=p.tau_E, tau_I=p.tau_I,
                             E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                             C_PV=p.C_PV)
    pv_neurons = NeuronGroup(p.N_pv, model=pv_equations, threshold='v > V_t',
                             reset='v = E_l', refractory=8.3 * ms, method='euler')
    pv_neurons.v = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to PV neurons
    for n_idx in range(p.N_pv):
        pv_input_i = PoissonInput(pv_neurons, 'g_e', N=1, rate=p.lambda_pv, weight=f'I_ext_pv(t, {n_idx})')

    # CS Neurons
    if use_dendrite_model:
        cs_equations_with_dendrite = Equations(eqs_exc_with_dendrite,
                                               tau_S=p.tau_S, tau_D=p.tau_D, tau_E=p.tau_E, tau_I=p.tau_I,
                                               E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                                               E_d=p.E_d, D_d=p.D_d,
                                               C_S=p.C_S, C_D=p.C_D,
                                               c_d=p.c_d, g_s=p.g_s, g_d=p.g_d
                                               )
        cs_neurons = NeuronGroup(p.N_cs, model=cs_equations_with_dendrite, threshold='v_s > V_t',
                                 reset='v_s = E_l', refractory=8.3 * ms, method='euler')
        cs_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
        cs_neurons.v_d = -70 * mV
    else:
        cs_equations_without_dendrite = Equations(eqs_exc_without_dendrite,
                                                  tau_S=p.tau_S, tau_E=p.tau_E, tau_I=p.tau_I,
                                                  E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                                                  C_S=p.C_S)
        cs_neurons = NeuronGroup(p.N_cs, model=cs_equations_without_dendrite, threshold='v_s > V_t',
                                 reset='v_s = E_l', refractory=8.3 * ms, method='euler')
        cs_neurons.v_s = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to CS neurons
    for n_idx in range(p.N_cs):
        cs_input_i = PoissonInput(cs_neurons, 'g_es', N=1, rate=p.lambda_cs, weight=f'I_ext_cs(t, {n_idx})')

    # CC Neurons
    if use_dendrite_model:
        cc_equations_with_dendrite = Equations(eqs_exc_with_dendrite,
                                               tau_S=p.tau_S, tau_D=p.tau_D, tau_E=p.tau_E, tau_I=p.tau_I,
                                               E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                                               E_d=p.E_d, D_d=p.D_d,
                                               C_S=p.C_S, C_D=p.C_D,
                                               c_d=p.c_d, g_s=p.g_s, g_d=p.g_d
                                               )
        cc_neurons = NeuronGroup(p.N_cc, model=cc_equations_with_dendrite, threshold='v_s > V_t',
                                 reset='v_s = E_l', refractory=8.3 * ms, method='euler')
        cc_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
        cc_neurons.v_d = -70 * mV
    else:
        cc_equations_without_dendrite = Equations(eqs_exc_without_dendrite,
                                                  tau_S=p.tau_S, tau_E=p.tau_E, tau_I=p.tau_I,
                                                  E_l=p.E_l, E_e=p.E_e, E_i=p.E_i,
                                                  C_S=p.C_S)
        cc_neurons = NeuronGroup(p.N_cc, model=cc_equations_without_dendrite, threshold='v_s > V_t',
                                 reset='v_s = E_l', refractory=8.3 * ms, method='euler')
        cc_neurons.v_s = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to CC neurons
    for n_idx in range(p.N_cc):
        cc_input_i = PoissonInput(cc_neurons, 'g_es', N=1, rate=p.lambda_cc, weight=f'I_ext_cc(t, {n_idx})')

    neurons = [sst_neurons, pv_neurons, cs_neurons, cc_neurons]

    # ##############################################################################
    # Define Synapses (common synapses for simulation)
    # ##############################################################################

    synapses = {}
    # SST <=> PV
    conn_SST_PV = Synapses(sst_neurons, pv_neurons, model='w: 1', on_pre='g_i+=w*nS', name='SST_PV')  # inhibitory
    conn_SST_PV.connect(p=p.pSST_PV if use_synaptic_probabilities else 1)
    conn_SST_PV.w = p.wSST_PV
    synapses["SST_PV"] = conn_SST_PV

    conn_PV_SST = Synapses(pv_neurons, sst_neurons, model='w: 1', on_pre='g_i+=w*nS', name='PV_SST')  # inhibitory
    conn_PV_SST.connect(p=p.pPV_SST if use_synaptic_probabilities else 1)
    conn_PV_SST.w = p.wPV_SST
    synapses["PV_SST"] = conn_PV_SST

    # PV <=> PYR soma
    ## target CS soma
    conn_PV_CSsoma = Synapses(pv_neurons, cs_neurons, model='w: 1', on_pre='g_is+=w*nS', name='PV_CSsoma')  # inhibitory
    conn_PV_CSsoma.connect(p=p.pPV_CS if use_synaptic_probabilities else 1)
    conn_PV_CSsoma.w = p.wPV_CS
    synapses["PV_CSsoma"] = conn_PV_CSsoma

    conn_CSsoma_PV = Synapses(cs_neurons, pv_neurons, model='w: 1', on_pre='g_e+=w*nS', name='CSsoma_PV')  # excitatory
    conn_CSsoma_PV.connect(p=p.pCS_PV if use_synaptic_probabilities else 1)
    conn_CSsoma_PV.w = p.wCS_PV
    synapses["CSsoma_PV"] = conn_CSsoma_PV

    ## target CC soma
    conn_PV_CCsoma = Synapses(pv_neurons, cc_neurons, model='w: 1', on_pre='g_is+=w*nS', name='PV_CCsoma')  # inhibitory
    conn_PV_CCsoma.connect(p=p.pPV_CC if use_synaptic_probabilities else 1)
    conn_PV_CCsoma.w = p.wPV_CC
    synapses["PV_CCsoma"] = conn_PV_CCsoma

    conn_CCsoma_PV = Synapses(cc_neurons, pv_neurons, model='w: 1', on_pre='g_e+=w*nS', name='CCsoma_PV')  # excitatory
    conn_CCsoma_PV.connect(p=p.pCC_PV if use_synaptic_probabilities else 1)
    conn_CCsoma_PV.w = p.wCC_PV
    synapses["CCsoma_PV"] = conn_CCsoma_PV

    # PYR => SST soma
    conn_CSsoma_SST = Synapses(cs_neurons, sst_neurons, model='w: 1', on_pre='g_e+=w*nS',
                               name='CSsoma_SST')  # excitatory
    conn_CSsoma_SST.connect(p=p.pCS_SST if use_synaptic_probabilities else 1)
    conn_CSsoma_SST.w = p.wCS_SST
    synapses["CSsoma_SST"] = conn_CSsoma_SST

    ## taget CC soma
    conn_CCsoma_SST = Synapses(cc_neurons, sst_neurons, model='w: 1', on_pre='g_e+=w*nS',
                               name='CCsoma_SST')  # excitatory
    conn_CCsoma_SST.connect(p=p.pCC_SST if use_synaptic_probabilities else 1)
    conn_CCsoma_SST.w = p.wCC_SST
    synapses["CCsoma_SST"] = conn_CCsoma_SST

    # CC => CS
    ## target CS soma
    conn_CCsoma_CSsoma = Synapses(cc_neurons, cs_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CC_CSsoma')  # excitatory
    conn_CCsoma_CSsoma.connect(p=p.pCC_CS if use_synaptic_probabilities else 1)
    conn_CCsoma_CSsoma.w = p.wCC_CS
    synapses["CCsoma_CSsoma"] = conn_CCsoma_CSsoma

    # self connections
    ## CS soma self connection
    conn_CSsoma_CSsoma = Synapses(cs_neurons, cs_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CSsoma_CSsoma')  # excitatory
    conn_CSsoma_CSsoma.connect(p=p.pCS_CS if use_synaptic_probabilities else 1)
    conn_CSsoma_CSsoma.w = p.wCS_CS
    synapses["CSsoma_CSsoma"] = conn_CSsoma_CSsoma

    backprop_CS = Synapses(cs_neurons, cs_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CS')
    backprop_CS.connect(condition='i==j')  # Connect all CS neurons to themselves

    ## CC soma self connection
    conn_CCsoma_CCsoma = Synapses(cc_neurons, cc_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CCsoma_CCsoma')  # excitatory
    conn_CCsoma_CCsoma.connect(p=p.pCC_CC if use_synaptic_probabilities else 1)
    conn_CCsoma_CCsoma.w = p.wCC_CC
    synapses["CCsoma_CCsoma"] = conn_CCsoma_CCsoma

    backprop_CC = Synapses(cc_neurons, cc_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CC')
    backprop_CC.connect(condition='i==j')  # Connect all CC neurons to themselves

    ## SST self connection
    conn_SST_SST = Synapses(sst_neurons, sst_neurons, model='w: 1', on_pre='g_i+=w*nS', name='SST_SST')  # inhibitory
    conn_SST_SST.connect(p=p.pSST_SST if use_synaptic_probabilities else 1)
    conn_SST_SST.w = p.wSST_SST
    synapses["SST_SST"] = conn_SST_SST

    ## PV self connection
    conn_PV_PV = Synapses(pv_neurons, pv_neurons, model='w: 1', on_pre='g_i+=w*nS', name='PV_PV')  # inhibitory
    conn_PV_PV.connect(p=p.pPV_PV if use_synaptic_probabilities else 1)
    conn_PV_PV.w = p.wPV_PV
    synapses["PV_PV"] = conn_PV_PV

    network = Network(collect())
    network.store('initialized')

    defaultclock.dt = p.sim_dt

    # ##############################################################################
    # Continue defining specific simulation run
    # ##############################################################################

    results = []
    if use_dendrite_model:
        for p_SST_CS_soma, p_SST_CC_soma in zip(p.pSST_CS_soma, p.pSST_CC_soma):
            result = run_simulation_for_weighted_sst_soma_dendrite(network, neurons, synapses, p,
                                                                   p_SST_CS_soma, p_SST_CC_soma,
                                                                   base_output_folder,
                                                                   use_synaptic_probabilities)

            result["pSST_CS_soma"] = p_SST_CS_soma
            result["pSST_CC_soma"] = p_SST_CC_soma
            results.append(result)
    else:
        result = run_simulation_without_exh_dendrite(network, neurons, synapses, p, base_output_folder, use_synaptic_probabilities)

        result["pSST_CS_soma"] = 1
        result["pSST_CC_soma"] = 1
        results.append(result)

    return results


def run_complete_simulation(params, use_dendrite_model=True, use_synaptic_probabilities=True, seed_val=12345):
    """
    Runs a complete simulation for given parameters.
    Computes both individual simulation results and aggregated simulation results.
    Selectivity of network is calculated through varying external input to neurons.
    """

    p = Struct(**params)

    N = [p.N_cs, p.N_cc, p.N_sst, p.N_pv]
    degrees = p.degrees

    input_steady = [p.I_cs_steady, p.I_cc_steady, p.I_sst_steady, p.I_pv_steady]
    input_amplitudes = [p.I_cs_amp, p.I_cc_amp, p.I_sst_amp, p.I_pv_amp]

    length = np.random.uniform(0, 1, (np.sum(N),))
    angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
    a_data = np.sqrt(length) * np.cos(angle)
    b_data = np.sqrt(length) * np.sin(angle)

    spatial_F = 10
    spatial_phase = 1
    tsteps = int(p.duration / p.sim_dt)

    assert len(p.pSST_CS_soma) == len(p.pSST_CC_soma)

    if use_dendrite_model:
        degree_results_vector = [[] for _ in p.pSST_CS_soma]
    else:
        degree_results_vector = [[]]

    ################## iterate through different input angles ##################
    for degree in degrees:
        print(f"Running simulations for input of degree {degree} ...")
        rad = math.radians(degree)
        inputs = hlp.distributionInput(
            a_data=a_data, b_data=b_data,
            spatialF=spatial_F, orientation=rad,
            spatialPhase=spatial_phase, amplitude=input_amplitudes, T=tsteps,
            steady_input=input_steady, N=N
        )

        params_with_input = copy.copy(params)

        params_with_input["I_ext_cs"] = inputs[:, :p.N_cs]
        params_with_input["I_ext_cc"] = inputs[:, p.N_cs:p.N_cs+p.N_cc]
        params_with_input["I_ext_sst"] = inputs[:, p.N_cs+p.N_cc:p.N_cs+p.N_cc+p.N_sst]
        params_with_input["I_ext_pv"] = inputs[:, p.N_cs+p.N_cc+p.N_sst:]

        results = run_simulation_for_input(params_with_input, seed_val=seed_val,
                                 use_synaptic_probabilities=use_synaptic_probabilities,
                                 use_dendrite_model=use_dendrite_model,
                                 base_output_folder=f'output/{degree}')


        for idx, result in enumerate(results):
            pSST_CS_soma = result["pSST_CS_soma"]
            pSST_CC_soma = result["pSST_CC_soma"]

            degree_results_vector[idx].append(result)
            hlp.save_results_to_folder(result,
                                       output_folder=f'output/{degree}/sst_soma_{pSST_CS_soma}CS_{pSST_CC_soma}CC',
                                       file_name='results.json')

    ################## calculate aggregate statistics for previous simulations ##################
    agg_results_vector = []
    for degree_results in degree_results_vector:
        pSST_CS_soma = degree_results[0]["pSST_CS_soma"]
        pSST_CC_soma = degree_results[0]["pSST_CC_soma"]

        agg_results = hlp.calculate_aggregate_results(degree_results)
        agg_results["pSST_CS_soma"] = pSST_CS_soma
        agg_results["pSST_CC_soma"] = pSST_CC_soma

        agg_results_vector.append(agg_results)
        hlp.save_agg_results_to_folder(agg_results,
                                       output_folder='output',
                                       file_name=f'agg_results_sst_soma_{pSST_CS_soma}CS_{pSST_CC_soma}CC.json')

    plot_selectivity_comparison(agg_results_vector, output_folder='output')
