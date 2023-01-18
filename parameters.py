from brian2 import *

N_SST = 34
N_PV = 46
N_CC = 275
N_CS = 45

default = {
    ### General parameters
    "duration": 5 * second,  # Total simulation time
    "sim_dt":   0.1 * ms,    # Integrator/sampling step

    "N_sst": N_SST,  # Number of SST neurons (inhibitory)
    "N_pv":  N_PV,   # Number of PV neurons (inhibitory)
    "N_cc":  N_CC,   # Number of CC (PYR) neurons (excitatory)
    "N_cs":  N_CS,   # Number of CS (PYR) neurons (excitatory)

    ### Neuron parameters
    "tau_S":    16 * ms,   # PYR neuron - soma membrane time constant
    "tau_D":     7 * ms,   # PYR neuron - dendritic membrane time constant
    "tau_SST":  20 * ms,   # SST neuron membrane time constant
    "tau_PV":   10 * ms,   # PV neuron membrane time constant
    "tau_E":     5 * ms,   # Excitatory synaptic time constant
    "tau_I":    10 * ms,   # Inhibitory synaptic time constant

    "C_S":      370 * pF,  # PYR neuron - soma membrane capacitance
    "C_D":      170 * pF,  # PYR neuron - dendritic membrane capacitance
    "C_SST":    100 * pF,  # SST neuron membrane capacitance
    "C_PV":     100 * pF,  # PV neuron membrane capacitance

    "E_l":  -70 * mV,  # leak reversal potential
    "E_e":    0 * mV,  # Excitatory synaptic reversal potential
    "E_i":  -80 * mV,  # Inhibitory synaptic reversal potential

    "V_t":  -50 * mV,  # spiking threshold

    "c_d": 2600 * pA,  # back-propagates somatic spikes to the dendrites
    "g_s": 1300 * pA,  # propagates dendritic regenerative activity to soma
    "g_d": 1200 * pA,  # propagates dendritic regenerative activity to dendrites

    ### Sigmoid function params
    "E_d":  -38 * mV,  # position control of threshold
    "D_d":    6 * mV,  # sharpness control of threshold

    ### Synpatic connection weights & probabilities
    "wCS_CS":    0.27,
    "pCS_CS":    0.16,

    "wCS_SST":   0.05,
    "pCS_SST":   0.23,

    "wCS_PV":    1.01,
    "pCS_PV":    0.18,

    "wSST_CS":        0.19,
    "pSST_CS":        0.52,
    "pSST_CS_weight": 1,    # represents fraction of `pSST_CS` for connection probabilities
    "pSST_CS_soma":   [0.5, 0.5,   1,  1],  # represents fraction of `pSST_CS*pSST_CS_weight` going to the CS soma; {1-this} goes to CS dendrite
    # `pSST_CS_soma` and `pSST_CC_soma` are taken in pairs
    "pSST_CC_soma":   [0.5,   1, 0.5,  1],   # represents fraction of `pSST_CC*pSST_CC_weight` going to the CC soma;  {1-this} goes to CC dendrite
    "wSST_CC": 0.11,
    "pSST_CC": 0.13,
    "pSST_CC_weight": 1,  # represents fraction of `pSST_CC` for connection probabilities

    "wPV_CS":    0.32,
    "pPV_CS":    0.43,

    "wCC_CC":    0.24,
    "pCC_CC":    0.06,

    "wCC_SST":   0.09,
    "pCC_SST":   0.26,

    "wCC_PV":    0.48,
    "pCC_PV":    0.22,

    "wPV_CC":    0.52,
    "pPV_CC":    0.38,

    "wCC_CS":    0.19,
    "pCC_CS":    0.09,

    "wSST_PV":   0.18,
    "pSST_PV":   0.29,

    "wSST_SST":  0.19,
    "pSST_SST":  0.1,

    "wPV_PV":    0.47,
    "pPV_PV":    0.5,

    "wPV_SST":   0.44,
    "pPV_SST":   0.14,

    ### Input amplitude & steady state
    "I_sst_amp":    50,
    "I_pv_amp":     50,
    "I_cc_amp":     200,
    "I_cs_amp":     100,

    "I_sst_steady":  0,
    "I_pv_steady":   0,
    "I_cc_steady":   0,
    "I_cs_steady":   0,

    "lambda_cc":  10*Hz,
    "lambda_cs":  10*Hz,
    "lambda_sst": 10*Hz,
    "lambda_pv":  10*Hz,

    ### Degrees for simulated orientation input
    "degrees": [0, 90, 180, 270],


    ### Params for simulation analysis
    "no_bins_firing_rates":     10,  # Number of bins for firing rates historgram
    "no_bins_isi":              10,  # Number of bins for interspike intervals historgram

    "plot_connectivity_graph":      True,  # If true, will also plot synapse connectivity graph for each simulation
    "recompute_equilibrium":        False,  # If true, will try and recompute equilibirum time, if not will use `default_equilibrium_time`
    "default_equilibrium_t":        0.2 * second  # Default equilibirium time, will be used in case `recompute_equilibrium` is False. Should be set based on previous simulation results
}
