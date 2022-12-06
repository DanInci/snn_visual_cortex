from brian2 import *

N_SST = 34
N_PV = 46
N_CC = 275
N_CS = 45

default = {
    ### General parameters
    "duration": 3 * second,  # Total simulation time
    "sim_dt":   0.1 * ms,  # Integrator/sampling step

    "N_sst": N_SST,  # Number of SST neurons (inhibitory)
    "N_pv":  N_PV,  # Number of PV neurons (inhibitory)
    "N_cc":  N_CC,  # Number of CC neurons (excitatory)
    "N_cs":  N_CS,  # Number of CS neurons (excitatory)

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

    ### Synpatic connection weights & probabilities
    "wCS_CS":    0.27,
    "pCS_CS":    0.16,

    "wCS_SST":   0.05,
    "pCS_SST":   0.23,

    "wCS_PV":    1.01,
    "pCS_PV":    0.18,

    "wSST_CS":   0.19,
    "pSST_CS":   0.52,

    "wPV_CS":    0.32,
    "pPV_CS":    0.43,

    "wCC_CC":    0.24,
    "pCC_CC":    0.06,

    "wCC_SST":   0.09,
    "pCC_SST":   0.26,

    "wCC_PV":    0.48,
    "pCC_PV":    0.22,

    "wSST_CC":   0.19,
    "pSST_CC":   0.13,

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
    "degrees": [0, 90, 180, 270]
}
