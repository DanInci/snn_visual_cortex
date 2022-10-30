from brian2 import *

default = {
    ### General parameters
    "duration": 10 * second,  # Total simulation time
    "sim_dt":   0.1 * ms,  # Integrator/sampling step

    "N_sst": 2,  # Number of SST neurons (inhibitory)
    "N_pv":  2,  # Number of PV neurons (inhibitory)
    "N_cc":  2,  # Number of CC neurons (excitatory)
    "N_cs":  2,  # Number of CS neurons (excitatory)

    ### Neuron parameters
    "tau_S":    16 * ms,
    "tau_D":    7 * ms,
    "tau_SST":  20 * ms,
    "tau_PV":   10 * ms,
    "tau_E":    5 * ms,  # Excitatory synaptic time constant
    "tau_I":    10 * ms,  # Inhibitory synaptic time constant

    "C_S":      370 * pF,
    "C_D":      170 * pF,
    "C_SST":    100 * pF,
    "C_PV":     100 * pF,

    "E_l":  -70 * mV,  # leak reversal potential
    "E_e":    0 * mV,  # Excitatory synaptic reversal potential
    "E_i":  -80 * mV,  # Inhibitory synaptic reversal potential

    "V_t":  -50 * mV,  # spiking threashold
    "V_r":  -70 * mV,  # reset potential ~ same as E_l ~

    "c_d": 2600 * pA,  # back-propagates somatic spikes to to the dendrites
    "g_s": 1300 * pA,  # propagates dendritic regenerative activity to soma
    "g_d": 1200 * pA,  # propagates dendritic regenerative activity to denderites

    ### Sigmoid function params
    "E_d":  -38 * mV,  # position control of threshold
    "D_d":    6 * mV,  # sharpness control of threshold

    ### Synapse parameters
    "w_e": 0.05 * nS,  # Excitatory synaptic conductance
    "w_i":  1.0 * nS,  # Inhibitory synaptic conductance

    ### Connection probabilities
    "pCS_CS":    1,  #0.16,
    "pCS_SST":   1,  #0.23,
    "pCS_PV":    1,  #0.18,
    "pSST_CS":   1,  #0.52,
    "pPV_CS":    1,  #0.43,
    "pCC_CC":    1,  #0.06,
    "pCC_SST":   1,  #0.26,
    "pCC_PV":    1,  #0.22,
    "pSST_CC":   1,  #0.13,
    "pPV_CC":    1,  #0.38,
    "pCC_CS":    1,  #0.09,
    "pSST_PV":   1,  #0.29,
    "pSST_SST":  1,  #0.1,
    "pPV_PV":    1,  #0.5,
    "pPV_SST":   1,  #0.14,

    ### External Input
    "I_ext_sst":    [0*nS, 0*nS],
    "I_ext_pv":     [0*nS, 0*nS],
    "I_ext_cc":     [50*nS, 50*nS * 1.05],
    "I_ext_cs":     [50*nS, 50*nS * 1.05],

    "lambda_cc":  10*Hz,
    "lambda_cs":  10*Hz
}
