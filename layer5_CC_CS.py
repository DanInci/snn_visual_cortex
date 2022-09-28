import numpy as np
from brian2 import *

seed(12345)


def sigmoid(x):
    E_d = -38 * mV  # position control of threshold
    D_d = 6 * mV  # sharpness control of threshold

    return 1/(1+np.exp(-(-x-E_d)/D_d))


def run_simulation(params=None):
    # TODO override params
    start_scope()

    ################################################################################
    # Model parameters
    ################################################################################
    ### General parameters
    duration = 1.0 * second  # Total simulation time

    N_sst = 2  # Number of SST neurons (inhibitory)
    N_pv = 2  # Number of PV neurons (inhibitory)
    N_cc = 2  # Number of CC neurons (excitatory)
    N_cs = 2  # Number of CS neurons (excitatory)

    ### Neuron parameters
    tau_S = 16 * ms
    tau_D = 7 * ms
    tau_SST = 20 * ms
    tau_PV = 10 * ms

    C_S = 370 * pF
    C_D = 170 * pF
    C_SST = 100 * pF
    C_PV = 100 * pF

    E_L = -70 * mV  # leak reversal potential
    V_t = -50 * mV  # spiking threashold
    V_r = E_L  # reset potential
    c_d = 2600 * pA  #
    g_s = 1300 * pA  #
    g_d = 1200 * pA  #

    g = 1 * pA
    M = 1.05
    I_ext_sst = [g, g * M]  # external input SST
    I_ext_pv = [0 * pA, 0 * pA]  # external input PV
    ################################################################################


    ################################################################################
    # Define neurons & their equations
    ################################################################################

    # Equations for SST (inhibitory) neurons
    eqs_sst_inh = '''
        dv/dt=(-(v-E_L)/tau_SST + I/C_SST) : volt (unless refractory)
        I = I_external + I_syn : amp
        I_external : amp
        I_syn: amp
    '''

    # Equations for PV (inhibitory) neurons
    eqs_pv_inh = '''
        dv/dt=(-(v-E_L)/tau_PV + I/C_PV) : volt (unless refractory)
        I = I_external + I_syn : amp
        I_external : amp
        I_syn : amp
    '''

    # Equations for PYR (excitatory) neurons
    eqs_exc = '''
        dv_s/dt = (-(v_s-E_L)/tau_S + (g_s*sigmoid(v_d) + I_s)/C_S) : volt (unless refractory)
        I_s = I_external + I_syn_s : amp
        I_external : amp
        I_syn_s : amp

        dv_d/dt = (-(v_d-E_L)/tau_D + (g_d*sigmoid(v_d) + c_d*K + I_d)) : volt (unless refractory)
        I_d = I_syn_d : amp
        I_syn_d : amp
        K : 1
    '''

    # SST Neurons
    sst_neurons = NeuronGroup(N_sst, model=eqs_sst_inh, threshold='v > V_t',
                              reset='v = E_L', refractory=5 * ms, method='euler')
    sst_neurons.set_states({'I_external': I_ext_sst})  # init external input to SST Neurons
    sst_neurons.v = 'E_L + rand()*(V_t-E_L)'  # random init of potentials

    # PV Neurons
    pv_neurons = NeuronGroup(N_pv, model=eqs_pv_inh, threshold='v > V_t',
                             reset='v = E_L', refractory=5 * ms, method='euler')
    pv_neurons.set_states({'I_external': I_ext_pv})  # init external input to SST Neurons
    pv_neurons.v = 'E_L + rand()*(V_t-E_L)'  # random init of potentials

    # CS Neurons
    cs_neurons = NeuronGroup(N_cs, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_L', refractory=5 * ms, method='euler')
    cs_neurons.v_s = 'E_L + rand()*(V_t-E_L)'  # random init of potentials for soma compartment
    cs_neurons.v_d = 'E_L + rand()*(V_t-E_L)'  # random init of potentials for dendrite compartment

    # CC Neurons
    cc_neurons = NeuronGroup(N_cc, model=eqs_exc, threshold='v_s > V_t',
                             reset='v_s = E_L', refractory=5 * ms, method='euler')
    cc_neurons.v_s = 'E_L + rand()*(V_t-E_L)'  # random init of potentials for soma compartment
    cc_neurons.v_d = 'E_L + rand()*(V_t-E_L)'  # random init of potentials for dendrite compartment


    # ##############################################################################
    # # Define synapses & their equations
    # ##############################################################################

    # TODO synapses

    # ##############################################################################
    # # Monitors
    # ##############################################################################

    # TODO Monitors

    # ##############################################################################
    # # Simulation run
    # ##############################################################################
    run(duration, report='text')


run_simulation()