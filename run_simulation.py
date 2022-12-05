from layer5_CC_CS import run_complete_simulation
from parameters import default as default_params

run_complete_simulation(default_params, simulate_sst_target_soma=True, seed_val=12345)
