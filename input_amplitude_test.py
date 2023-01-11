from parameters import default as default_params
from layer5_CC_CS import run_simulation_for_input, Struct
from datetime import datetime

import numpy as np
import helpers as hlp
import copy
import math
import csv
import os


def run_input_amplitude_simulation(params, input_amplitudes, seed_val=12345):
    p = Struct(**params)

    N = [p.N_cs, p.N_cc, p.N_sst, p.N_pv]
    degrees = p.degrees

    input_steady = [p.I_cs_steady, p.I_cc_steady, p.I_sst_steady, p.I_pv_steady]

    length = np.random.uniform(0, 1, (np.sum(N),))
    angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
    a_data = np.sqrt(length) * np.cos(angle)
    b_data = np.sqrt(length) * np.sin(angle)

    spatial_F = 10
    spatial_phase = 1
    tsteps = int(p.duration / p.sim_dt)

    ################## iterate through different input angles ##################
    results_without_sst_soma = []
    for degree in degrees:
        rad = math.radians(degree)
        inputs = hlp.distributionInput(
            a_data=a_data, b_data=b_data,
            spatialF=spatial_F, orientation=rad,
            spatialPhase=spatial_phase, amplitude=input_amplitudes, T=tsteps,
            steady_input=input_steady, N=N
        )

        params_with_input = copy.copy(params)

        params_with_input["I_ext_cs"] = inputs[:, :p.N_cs]
        params_with_input["I_ext_cc"] = inputs[:, p.N_cs:p.N_cs + p.N_cc]
        params_with_input["I_ext_sst"] = inputs[:, p.N_cs + p.N_cc:p.N_cs + p.N_cc + p.N_sst]
        params_with_input["I_ext_pv"] = inputs[:, p.N_cs + p.N_cc + p.N_sst:]

        # Only simulate for SST -> SOMA connection NOT present
        params_with_input["pSST_CS_soma"] = [0]  # all probability goes to CS dendrite
        params_with_input["pSST_CC_soma"] = [0]  # all probability goes to CC dendrite

        results = run_simulation_for_input(params_with_input, seed_val=seed_val, use_synaptic_probabilities=True, use_dendrite_model=True)

        assert len(results) == 1
        result_without_sst_soma = results[0]
        results_without_sst_soma.append(result_without_sst_soma)


    ################## calculate aggregate statistics for previous simulations ##################
    # Only simulated for SST -> SOMA connection NOT present
    agg_results_without_sst_to_soma = hlp.calculate_aggregate_results(results_without_sst_soma)

    return agg_results_without_sst_to_soma


def test_input_amplitudes(params, csv_writer=None, seed_val=12345):
    input_cs_amplitudes = [100]
    input_cc_amplitudes = [100, 200, 300]
    input_sst_amplitudes = [10, 25, 50, 75, 100]
    input_pv_amplitudes = [50]

    for input_cs in input_cs_amplitudes:
        for input_cc in input_cc_amplitudes:
            for input_sst in input_sst_amplitudes:
                for input_pv in input_pv_amplitudes:
                    input_amplitudes = [input_cs, input_cc, input_sst, input_pv]
                    print(f"Running simulations for input amplitudes {input_amplitudes} ...")

                    results = run_input_amplitude_simulation(params, input_amplitudes, seed_val)

                    row = [input_cs, input_cc, input_sst, input_pv,
                           results["mean_fire_rate_cs"], results["mean_fire_rate_cc"], results["mean_fire_rate_sst"], results["mean_fire_rate_pv"],
                           results.get("os_rel"), results.get("ds_rel"), results.get("os_paper_rel")]

                    # write into csv file
                    if csv_writer:
                        csv_writer.writerow(row)


now = datetime.now()
sim_name = 'input_amplitude_test'
time_id = now.strftime("%m:%d:%Y_%H:%M:%S")
output_folder = 'data'
output_file = f'{output_folder}/{sim_name}_{time_id}.csv'

header_row = ['input_cs', 'input_cc', 'input_sst', 'input_pv',
              'fire_rate_cs', 'fire_rate_cc', 'fire_rate_sst', 'fire_rate_pv',
              'os_rel', 'ds_rel', 'os_paper_rel']

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(output_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header_row)

    test_input_amplitudes(default_params, csv_writer=writer, seed_val=12345)



