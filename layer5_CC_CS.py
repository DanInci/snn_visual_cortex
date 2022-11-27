from brian2 import *
from plotting import *
from equations import *
from parameters import default as default_params
import copy

import helpers as hlp


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def analyse_network_simulation(spike_monitors, state_monitors, connections, V_t, sim_duration, output_folder=None):
    spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc = spike_monitors
    state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc = state_monitors

    ### General analysis parameters
    time_frame = 0.1  # Time frame for computing equlibrium time
    no_bins_firing_rates = 10 # Number of bins for firing rates historgram
    no_bins_isi = 10  # Number of bins for interspike intervals historgram

    plot_only_from_equilibrium = True  # Plot graphs only from equilibrium time
    recompute_equilibrium = False  # If true, will try and recompute equilibirum time, if not will use `default_equilibrium_time`
    default_equilibrium_t = 5*second  # Default equilibirium time, will be used in case `recompute_equilibrium` is False. Should be set based on previous simulation results

    ################################################################################
    # Analysis and plotting
    ################################################################################

    if recompute_equilibrium:
        equilibrium_times = []
        for idx, spike_mon in enumerate([spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv]):
            t, firing_rate = hlp.compute_equilibrium_for_neuron_type(spike_mon, time_frame)
            if firing_rate is not None:
                print(f"Found for {index_to_ntype_dict[idx]} neurons")

            equilibrium_times.append(t)

        equilibrium_t = max(equilibrium_times) * second
        if equilibrium_t < sim_duration:
            print(f"Equilibrium for all neurons start at: {equilibrium_t}")
        else:
            print(f"WARNING: Equilibrium was not found during the duration of the simulation")
    else:
        print(f"Skipping recalculating equilibrium time. Using default equilibrium time={default_equilibrium_t}")
        equilibrium_t = default_equilibrium_t

    # Only compute properties of the system from equilibrium time to end simulation time
    from_t = equilibrium_t
    to_t = sim_duration

    print("Plotting results from equilibrium ... ")

    raster_from_t = from_t if plot_only_from_equilibrium else 0
    raster_to_t = min(raster_from_t + 3 * second, sim_duration)
    plot_raster(spike_mon_cs, spike_mon_cc, spike_mon_sst, spike_mon_pv, raster_from_t, raster_to_t,
                output_folder=output_folder, file_name='spike_raster_plot')

    plot_states(state_mon_cs, spike_mon_cs, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder=output_folder,
                file_name='state_plot_CS')
    plot_states(state_mon_cc, spike_mon_cc, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder=output_folder,
                file_name='state_plot_CC')
    plot_states(state_mon_sst, spike_mon_sst, V_t, plot_only_from_equilibrium, from_t, to_t,
                output_folder=output_folder, file_name='state_plot_SST')
    plot_states(state_mon_pv, spike_mon_pv, V_t, plot_only_from_equilibrium, from_t, to_t, output_folder=output_folder,
                file_name='state_plot_PV')

    # Plot connectivity graph
    # plot_neuron_connectivity(connections, output_folder=output_folder, file_name='neuron_connectivity')

    results = {}

    # Compute firing rate for each neuron group

    results["firing_rates_cs"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cs, from_t, to_t)
    results["firing_rates_cc"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_cc, from_t, to_t)
    results["firing_rates_sst"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_sst, from_t, to_t)
    results["firing_rates_pv"] = hlp.compute_firing_rate_for_neuron_type(spike_mon_pv, from_t, to_t)

    firing_rates = [results["firing_rates_cs"], results["firing_rates_cc"], results["firing_rates_sst"],
                    results["firing_rates_pv"]]
    plot_firing_rate_histograms(firing_rates, no_bins_firing_rates, output_folder=output_folder,
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
    autocorr_cs = hlp.compute_autocorr_struct(results["interspike_intervals_cs"], no_bins_isi)
    if autocorr_cs:
        results["acorr_min_cs"] = autocorr_cs["minimum"]

    # for CC
    autocorr_cc = hlp.compute_autocorr_struct(results["interspike_intervals_cc"], no_bins_isi)
    if autocorr_cc:
        results["acorr_min_cc"] = autocorr_cc["minimum"]

    # for SST
    autocorr_sst = hlp.compute_autocorr_struct(results["interspike_intervals_sst"], no_bins_isi)
    if autocorr_sst:
        results["acorr_min_sst"] = autocorr_sst["minimum"]

    # for PV
    autocorr_pv = hlp.compute_autocorr_struct(results["interspike_intervals_pv"], no_bins_isi)
    if autocorr_pv:
        results["acorr_min_sst"] = autocorr_pv["minimum"]

    interspike_intervals = [results["interspike_intervals_cs"], results["interspike_intervals_cc"],
                            results["interspike_intervals_sst"], results["interspike_intervals_pv"]]
    autocorr = [autocorr_cs, autocorr_cc, autocorr_sst, autocorr_pv]
    plot_isi_histograms(interspike_intervals, no_bins_isi, autocorr=autocorr, output_folder=output_folder,
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


def run_simulation_for_input(params, seed_val=12345, simulate_sst_target_soma=False, use_synaptic_probabilities=True, base_output_folder=None):
    p = Struct(**params)

    start_scope()
    seed(seed_val)

    ################################################################################
    # Model parameters
    ################################################################################

    sim_duration = p.duration  # Total simulation time
    sim_dt = p.sim_dt  # Integrator/sampling step

    ### Network parameters
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

    ################################################################################

    ### External Input parameters
    I_ext_sst = TimedArray(p.I_ext_sst*nS, dt=sim_dt)
    I_ext_pv = TimedArray(p.I_ext_pv*nS, dt=sim_dt)
    I_ext_cs = TimedArray(p.I_ext_cs*nS, dt=sim_dt)
    I_ext_cc = TimedArray(p.I_ext_cc*nS, dt=sim_dt)

    lambda_sst = p.lambda_sst
    lambda_pv = p.lambda_pv
    lambda_cs = p.lambda_cs
    lambda_cc = p.lambda_cc

    N_sst = p.N_sst  # Number of SST neurons (inhibitory)
    N_pv = p.N_pv  # Number of PV neurons (inhibitory)
    N_cc = p.N_cc  # Number of CC neurons (excitatory)
    N_cs = p.N_cs  # Number of CS neurons (excitatory)

    ################################################################################
    # Define neurons
    ################################################################################

    # SST Neurons
    sst_neurons = NeuronGroup(N_sst, model=eqs_sst_inh, threshold='v > V_t',
                              reset='v = E_l', refractory=8.3 * ms, method='euler')
    sst_neurons.v = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to SST neurons
    for n_idx in range(N_sst):
        sst_input_i = PoissonInput(sst_neurons, 'g_e', N=1, rate=lambda_sst, weight=f'I_ext_sst(t, {n_idx})')

    # PV Neurons
    pv_neurons = NeuronGroup(N_pv, model=eqs_pv_inh, threshold='v > V_t',
                             reset='v = E_l', refractory=8.3 * ms, method='euler')
    pv_neurons.v = 'E_l + rand()*(V_t-E_l)'

    ## Poisson input to PV neurons
    for n_idx in range(N_pv):
        pv_input_i = PoissonInput(pv_neurons, 'g_e', N=1, rate=lambda_pv, weight=f'I_ext_pv(t, {n_idx})')

    # CS Neurons
    cs_neurons = NeuronGroup(N_cs, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cs_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cs_neurons.v_d = -70 * mV

    ## Poisson input to CS neurons
    for n_idx in range(N_cs):
        cs_input_i = PoissonInput(cs_neurons, 'g_es', N=1, rate=lambda_cs, weight=f'I_ext_cs(t, {n_idx})')

    # CC Neurons
    cc_neurons = NeuronGroup(N_cc, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_l', refractory=8.3 * ms, method='euler')
    cc_neurons.v_s = 'E_l + rand()*(V_t-E_l)'
    cc_neurons.v_d = -70 * mV

    ## Poisson input to CC neurons
    for n_idx in range(N_cc):
        cc_input_i = PoissonInput(cc_neurons, 'g_es', N=1, rate=lambda_cc, weight=f'I_ext_cc(t, {n_idx})')

    # ##############################################################################
    # Define Synapses
    # ##############################################################################

    connections = {}
    # SST <=> PV
    conn_SST_PV = Synapses(sst_neurons, pv_neurons, model='w: 1', on_pre='g_i+=w*nS', name='SST_PV')  # inhibitory
    conn_SST_PV.connect(p=p.pSST_PV if use_synaptic_probabilities else 1)
    conn_SST_PV.w = p.wSST_PV
    connections["SST_PV"] = conn_SST_PV

    conn_PV_SST = Synapses(pv_neurons, sst_neurons, model='w: 1', on_pre='g_i+=w*nS', name='PV_SST')  # inhibitory
    conn_PV_SST.connect(p=p.pPV_SST if use_synaptic_probabilities else 1)
    conn_PV_SST.w = p.wPV_SST
    connections["PV_SST"] = conn_PV_SST

    # PV <=> PYR soma
    ## target CS soma
    conn_PV_CSsoma = Synapses(pv_neurons, cs_neurons, model='w: 1', on_pre='g_is+=w*nS', name='PV_CSsoma')  # inhibitory
    conn_PV_CSsoma.connect(p=p.pPV_CS if use_synaptic_probabilities else 1)
    conn_PV_CSsoma.w = p.wPV_CS
    connections["PV_CSsoma"] = conn_PV_CSsoma

    conn_CSsoma_PV = Synapses(cs_neurons, pv_neurons, model='w: 1', on_pre='g_e+=w*nS', name='CSsoma_PV')  # excitatory
    conn_CSsoma_PV.connect(p=p.pCS_PV if use_synaptic_probabilities else 1)
    conn_CSsoma_PV.w = p.wCS_PV
    connections["CSsoma_PV"] = conn_CSsoma_PV

    ## target CC soma
    conn_PV_CCsoma = Synapses(pv_neurons, cc_neurons, model='w: 1', on_pre='g_is+=w*nS', name='PV_CCsoma')  # inhibitory
    conn_PV_CCsoma.connect(p=p.pPV_CC if use_synaptic_probabilities else 1)
    conn_PV_CCsoma.w = p.wPV_CC
    connections["PV_CCsoma"] = conn_PV_CCsoma

    conn_CCsoma_PV = Synapses(cc_neurons, pv_neurons, model='w: 1', on_pre='g_e+=w*nS', name='CCsoma_PV')  # excitatory
    conn_CCsoma_PV.connect(p=p.pCC_PV if use_synaptic_probabilities else 1)
    conn_CCsoma_PV.w = p.wCC_PV
    connections["CCsoma_PV"] = conn_CCsoma_PV

    # PYR => SST soma
    conn_CSsoma_SST = Synapses(cs_neurons, sst_neurons, model='w: 1', on_pre='g_e+=w*nS',
                               name='CSsoma_SST')  # excitatory
    conn_CSsoma_SST.connect(p=p.pCS_SST if use_synaptic_probabilities else 1)
    conn_CSsoma_SST.w = p.wCS_SST
    connections["CSsoma_SST"] = conn_CSsoma_SST

    ## taget CC soma
    conn_CCsoma_SST = Synapses(cc_neurons, sst_neurons, model='w: 1', on_pre='g_e+=w*nS',
                               name='CCsoma_SST')  # excitatory
    conn_CCsoma_SST.connect(p=p.pCC_SST if use_synaptic_probabilities else 1)
    conn_CCsoma_SST.w = p.wCC_SST
    connections["CCsoma_SST"] = conn_CCsoma_SST

    # CC => CS
    ## target CS soma
    conn_CCsoma_CSsoma = Synapses(cc_neurons, cs_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CC_CSsoma')  # excitatory
    conn_CCsoma_CSsoma.connect(p=p.pCC_CS if use_synaptic_probabilities else 1)
    conn_CCsoma_CSsoma.w = p.wCC_CS
    connections["CCsoma_CSsoma"] = conn_CCsoma_CSsoma

    # self connections
    ## CS soma self connection
    conn_CSsoma_CSsoma = Synapses(cs_neurons, cs_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CSsoma_CSsoma')  # excitatory
    conn_CSsoma_CSsoma.connect(p=p.pCS_CS if use_synaptic_probabilities else 1)
    conn_CSsoma_CSsoma.w = p.wCS_CS
    connections["CSsoma_CSsoma"] = conn_CSsoma_CSsoma

    backprop_CS = Synapses(cs_neurons, cs_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CS')
    backprop_CS.connect(condition='i==j')  # Connect all CS neurons to themselves

    ## CC soma self connection
    conn_CCsoma_CCsoma = Synapses(cc_neurons, cc_neurons, model='w: 1', on_pre='g_es+=w*nS',
                                  name='CCsoma_CCsoma')  # excitatory
    conn_CCsoma_CCsoma.connect(p=p.pCC_CC if use_synaptic_probabilities else 1)
    conn_CCsoma_CCsoma.w = p.wCC_CC
    connections["CCsoma_CCsoma"] = conn_CCsoma_CCsoma

    backprop_CC = Synapses(cc_neurons, cc_neurons, on_pre={'up': 'K += 1', 'down': 'K -=1'},
                           delay={'up': 0.5 * ms, 'down': 2 * ms}, name='backprop_CC')
    backprop_CC.connect(condition='i==j')  # Connect all CC neurons to themselves

    ## SST self connection
    conn_SST_SST = Synapses(sst_neurons, sst_neurons, model='w: 1', on_pre='g_i+=w*nS', name='SST_SST')  # inhibitory
    conn_SST_SST.connect(p=p.pSST_SST if use_synaptic_probabilities else 1)
    conn_SST_SST.w = p.wSST_SST
    connections["SST_SST"] = conn_SST_SST

    ## PV self connection
    conn_PV_PV = Synapses(pv_neurons, pv_neurons, model='w: 1', on_pre='g_i+=w*nS', name='PV_PV')  # inhibitory
    conn_PV_PV.connect(p=p.pPV_PV if use_synaptic_probabilities else 1)
    conn_PV_PV.w = p.wPV_PV
    connections["PV_PV"] = conn_PV_PV

    network = Network(collect())
    network.store('initialized')

    defaultclock.dt = sim_dt

    # ##############################################################################
    # Add extra synapses (Without SST->Soma)
    # ##############################################################################

    # SST => PYR soma
    ## target CS dendrite
    conn_SST_CSdendrite = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                   name='SST_CSdendrite')  # inhibitory
    conn_SST_CSdendrite.connect(p=p.pSST_CS if use_synaptic_probabilities else 1)
    conn_SST_CSdendrite.w = p.wSST_CS
    connections["SST_CSdendrite"] = conn_SST_CSdendrite

    ## target CC dendrite
    conn_SST_CCdendrite = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                   name='SST_CCdendrite')  # inhibitory
    conn_SST_CCdendrite.connect(p=p.pSST_CC if use_synaptic_probabilities else 1)
    conn_SST_CCdendrite.w = p.wSST_CC
    connections["SST_CCdendrite"] = conn_SST_CCdendrite

    extra_connections = [conn_SST_CSdendrite, conn_SST_CCdendrite]

    # ##############################################################################
    # Define Monitors (Without SST->Soma)
    # ##############################################################################

    # Record spikes of different neuron groups
    spike_mon_sst = SpikeMonitor(sst_neurons)
    spike_mon_pv = SpikeMonitor(pv_neurons)
    spike_mon_cs = SpikeMonitor(cs_neurons)
    spike_mon_cc = SpikeMonitor(cc_neurons)

    spike_monitors = [spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc]

    # Record conductances and membrane potential of neuron ni
    state_mon_sst = StateMonitor(sst_neurons, ['v', 'g_e', 'g_i'], record=[0])
    state_mon_pv = StateMonitor(pv_neurons, ['v', 'g_e', 'g_i'], record=[0])
    state_mon_cs = StateMonitor(cs_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])
    state_mon_cc = StateMonitor(cc_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])

    state_monitors = [state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc]

    # ##############################################################################
    # Run Network (Without SST->Soma)
    # ##############################################################################

    print('* Run network simulation WITHOUT SST->Soma')
    network.restore('initialized')

    # Add extras to network
    network.add(extra_connections)
    network.add(spike_monitors)
    network.add(state_monitors)

    network.run(sim_duration, report='text')

    output_folder_without_sst_soma = f'{base_output_folder}/without_sst_soma' if base_output_folder else None
    results_without_sst_soma = analyse_network_simulation(spike_monitors, state_monitors, connections,
                                         V_t=V_t, sim_duration=sim_duration,
                                         output_folder=output_folder_without_sst_soma)

    # Cleanup extras from network
    network.remove(extra_connections)
    network.remove(spike_monitors)
    network.remove(state_monitors)

    results_with_sst_soma = None
    if simulate_sst_target_soma:
        # ##############################################################################
        # Add extra synapses (WithSST->Soma)
        # ##############################################################################

        # SST => PYR soma
        ## target CS soma
        conn_SST_CSsoma = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_is+=w*nS',
                                   name='SST_CSsoma')  # inhibitory (optional connection)
        conn_SST_CSsoma.connect(p=p.pSST_CS / 2 if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
        conn_SST_CSsoma.w = p.wSST_CS
        connections["SST_CSsoma"] = conn_SST_CSsoma

        ## target CS dendrite
        conn_SST_CSdendrite = Synapses(sst_neurons, cs_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                       name='SST_CSdendrite')  # inhibitory
        conn_SST_CSdendrite.connect(p=p.pSST_CS/2 if use_synaptic_probabilities else 1)
        conn_SST_CSdendrite.w = p.wSST_CS
        connections["SST_CSdendrite"] = conn_SST_CSdendrite

        ## target CC soma
        conn_SST_CCsoma = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_is+=w*nS',
                                   name='SST_CCsoma')  # inhibitory (optional connection)
        conn_SST_CCsoma.connect(p=p.pSST_CC / 2 if use_synaptic_probabilities else 1)  # inhibitory (optional connection)
        conn_SST_CCsoma.w = p.wSST_CC
        connections["SST_CCsoma"] = conn_SST_CCsoma

        ## target CC dendrite
        conn_SST_CCdendrite = Synapses(sst_neurons, cc_neurons, model='w: 1', on_pre='g_id+=w*nS',
                                       name='SST_CCdendrite')  # inhibitory
        conn_SST_CCdendrite.connect(p=p.pSST_CC/2 if use_synaptic_probabilities else 1)
        conn_SST_CCdendrite.w = p.wSST_CC
        connections["SST_CCdendrite"] = conn_SST_CCdendrite

        extra_connections = [conn_SST_CSsoma, conn_SST_CSdendrite, conn_SST_CCsoma, conn_SST_CCdendrite]

        # ##############################################################################
        # Define Monitors (With SST->Soma)
        # ##############################################################################

        # Record spikes of different neuron groups
        spike_mon_sst = SpikeMonitor(sst_neurons)
        spike_mon_pv = SpikeMonitor(pv_neurons)
        spike_mon_cs = SpikeMonitor(cs_neurons)
        spike_mon_cc = SpikeMonitor(cc_neurons)

        spike_monitors = [spike_mon_sst, spike_mon_pv, spike_mon_cs, spike_mon_cc]

        # Record conductances and membrane potential of neuron ni
        state_mon_sst = StateMonitor(sst_neurons, ['v', 'g_e', 'g_i'], record=[0])
        state_mon_pv = StateMonitor(pv_neurons, ['v', 'g_e', 'g_i'], record=[0])
        state_mon_cs = StateMonitor(cs_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])
        state_mon_cc = StateMonitor(cc_neurons, ['v_s', 'v_d', 'g_es', 'g_is', 'g_ed', 'g_id'], record=[0])

        state_monitors = [state_mon_sst, state_mon_pv, state_mon_cs, state_mon_cc]

        # ##############################################################################
        # Run Network (With SST->Soma)
        # ##############################################################################

        print('* Run network simulation WITH SST->Soma')
        network.restore('initialized')

        # Add extras to network
        network.add(extra_connections)
        network.add(spike_monitors)
        network.add(state_monitors)

        network.run(sim_duration, report='text')

        output_folder_with_sst_soma = f'{base_output_folder}/with_sst_soma' if base_output_folder else None
        results_with_sst_soma = analyse_network_simulation(spike_monitors, state_monitors, connections,
                                                              V_t=V_t, sim_duration=sim_duration,
                                                              output_folder=output_folder_with_sst_soma)
        # Cleanup extras from network
        network.remove(extra_connections)
        network.remove(spike_monitors)
        network.remove(state_monitors)

    return results_without_sst_soma, results_with_sst_soma


def simulate_with_different_inputs(params, simulate_sst_target_soma=True, seed_val=12345):
    p = Struct(**params)

    N = [p.N_cs, p.N_cc, p.N_sst, p.N_pv]
    degrees = p.degrees

    input_steady = [p.I_cs_steady, p.I_cc_steady, p.I_sst_steady, p.I_pv_steady]  # TODO What is this?
    input_amplitudes = [p.I_cs_amp, p.I_cc_amp, p.I_sst_amp, p.I_pv_amp]

    length = np.random.uniform(0, 1, (np.sum(N),))
    angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
    a_data = np.sqrt(length) * np.cos(angle)
    b_data = np.sqrt(length) * np.sin(angle)

    spatial_F = 10  # TODO What exactly does `spatial_F` do?
    temporal_F = 50  # TODO What exactly does `temporal_F` do?
    spatial_phase = 1  # TODO What exactly does `spatial_phase` do?
    tsteps = int(p.duration / p.sim_dt)

    ################## iterate through different inputs ##################
    results_with_sst_soma = []
    results_without_sst_soma = []
    for degree in degrees:
        print(f"Running simulations for input of degree {degree} ...")
        rad = math.radians(degree)
        inputs = hlp.distributionInput(
            a_data=a_data, b_data=b_data,
            spatialF=spatial_F, temporalF=temporal_F, orientation=rad,
            spatialPhase=spatial_phase, amplitude=input_amplitudes, T=tsteps,
            steady_input=input_steady, N=N
        )

        params_with_input = copy.copy(params)

        params_with_input["I_ext_cs"] = inputs[:, :p.N_cs]
        params_with_input["I_ext_cc"] = inputs[:, p.N_cs:p.N_cs+p.N_cc]
        params_with_input["I_ext_sst"] = inputs[:, p.N_cs+p.N_cc:p.N_cs+p.N_cc+p.N_sst]
        params_with_input["I_ext_pv"] = inputs[:, p.N_cs+p.N_cc+p.N_sst:]

        result_without_sst_soma, result_with_sst_soma = run_simulation_for_input(params_with_input, seed_val=seed_val,
                                 simulate_sst_target_soma=simulate_sst_target_soma, use_synaptic_probabilities=True,
                                 base_output_folder=f'output/{degree}')

        if result_without_sst_soma:
            results_without_sst_soma.append(result_without_sst_soma)
            hlp.save_results_to_folder(result_without_sst_soma, output_folder=f'output/{degree}/without_sst_soma', file_name='results.json')

        if result_with_sst_soma:
            results_with_sst_soma.append(result_with_sst_soma)
            hlp.save_results_to_folder(result_with_sst_soma, output_folder=f'output/{degree}/with_sst_soma', file_name='results.json')

    ################## calculate aggregate statistics for previous simulations ##################
    # SST -> SOMA connection NOT present
    agg_results_without_sst_to_soma = hlp.calculate_aggregate_results(results_without_sst_soma)
    hlp.save_agg_results_to_folder(agg_results_without_sst_to_soma,
                                   output_folder='output',
                                   file_name='agg_results_without_sst_soma.json')

    # SST -> SOMA connection present
    if simulate_sst_target_soma:
        agg_results_with_sst_to_soma = hlp.calculate_aggregate_results(results_with_sst_soma)
        hlp.save_agg_results_to_folder(agg_results_with_sst_to_soma,
                                       output_folder='output',
                                       file_name='agg_results_with_sst_soma.json')

        plot_selectivity_comparison(agg_results_with_sst=agg_results_with_sst_to_soma,
                                    agg_results_without_sst=agg_results_without_sst_to_soma, output_folder='output')


simulate_with_different_inputs(default_params, simulate_sst_target_soma=True, seed_val=12345)
