from brian2 import *


# TODO see how to reference this from equations
@check_units(x=volt, result=1)
def sigmoid(x):
    ### Sigmoid function params
    E_d = -38 * mV  # position control of threshold
    D_d = 6 * mV  # sharpness control of threshold

    return 1/(1+np.exp(-(-x-E_d)/D_d))