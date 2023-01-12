# Equations for SST (inhibitory) neurons
eqs_sst_inh = '''
    dv/dt = ((E_l-v)/tau_SST + I/C_SST) : volt (unless refractory)

    dg_e/dt = -g_e/tau_E : siemens
    dg_i/dt = -g_i/tau_I : siemens
    
    I = g_e*(E_e - v) + g_i*(E_i - v) : amp
'''

# Equations for PV (inhibitory) neurons
eqs_pv_inh = '''
    dv/dt = ((E_l-v)/tau_PV + I/C_PV) : volt (unless refractory)

    dg_e/dt = -g_e/tau_E : siemens
    dg_i/dt = -g_i/tau_I : siemens

    I = g_e*(E_e - v) + g_i*(E_i - v) : amp
'''

# Equations for PYR (excitatory) neurons WITH dendrites
eqs_exc_with_dendrite = '''
    dv_s/dt = ((E_l-v_s)/tau_S + (g_s*(1/(1+exp(-(v_d-E_d)/D_d))) + I_s)/C_S) : volt (unless refractory)

    dg_es/dt = -g_es/tau_E : siemens
    dg_is/dt = -g_is/tau_I : siemens

    I_s = g_es*(E_e - v_s) + g_is*(E_i - v_s) : amp

    dv_d/dt = ((E_l-v_d)/tau_D + (g_d*(1/(1+exp(-(v_d-E_d)/D_d))) + c_d*K + I_d)/C_D) : volt

    dg_ed/dt = -g_ed/tau_E : siemens
    dg_id/dt = -g_id/tau_I : siemens

    I_d = g_ed*(E_e - v_d) + g_id*(E_i - v_d) : amp
    K : 1
'''

# Equations for PYR (excitatory) neurons WITHOUT dendrites
eqs_exc_without_dendrite = '''
    dv_s/dt = ((E_l-v_s)/tau_S + I_s/C_S) : volt (unless refractory)

    dg_es/dt = -g_es/tau_E : siemens
    dg_is/dt = -g_is/tau_I : siemens

    I_s = g_es*(E_e - v_s) + g_is*(E_i - v_s) : amp
    K : 1
'''