Spiking Neural Network for the Visual Cortex (V1) Layer 5
============================
This is a spike neural network model of the V1, layer 5 includes Cortico-Cortical (CC), Cortico-Subcortical (CS), PV and SST neurons. 

Individual simulation metrics for neuron groups are computed such as firing rates, inter-spike intervals, burst detection.

Additionally, orientation and direction selectivity are aggregated for population of cells.

### Project content

    .
    ├── data                                        # Contains hypertuning results from amplitude simulation run
    ├── output                                      # Contains output results & plots from complete simulation run (.gitignored)
    ├── notebooks                                   # Jupyter notebooks used for testing different scenarios
    │   ├── layer5_CC_CS_connection.ipynb           # Scalar plot simulation for weighted CC->CS connection probability
    │   ├── layer5_SST_Soma_selectivity.ipynb       # Scalar plot simulation for weighted SST->CC/CS Soma connection probability
    │   ├── layer5_sandbox.ipynb                    # Sandbox notebook used when creating initial network topology
    │   └── ...               
    ├── layer5_CC_CS.py                             # Main network file that defines the topology of the simulation                   
    ├── equations.py                                # Equations for different neuron types
    ├── parameters.py                               # Default parameters for simulation
    ├── helpers.py                                  # Helper methods for analysis
    ├── plotting.py                                 # Helper methods for plotting
    ├── run_amplitude_hypertuing.py                 # Script for running multiple simulations for amplitude hypertuning for different neurons
    └── run_complete_simulation.py                  # Complete simulation run entrypoint

### Requirements
Use `conda env create -n network -f requirements.txt` to unpack the conda environment. Or download with `conda install <module>` the requirements manually. 

After that, activate the environment with `conda activate network`

### Hints
- There is only one best parameter set which I aggregated in `parameters.py`. If you want to try different combinations of parameters, you can change them here.
- To run main simulation `python3 run_complete_simulation.py`
- The project includes a script for hypertuning different input amplitudes for neuron groups. Can be run by `python3 run_amplitude_hypertuning.py`
- All simulation outputs are persisted in the `output` folder. I recommend deleting the content between successive runs to avoid confusion.
- Parallelization of simulations is missing. This would be a much needed improvement for the future.

### Project Report
- For more details and documentation about the project scope, the [report](./report/Project_Report.pdf) can be consulted.