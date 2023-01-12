Spiking Neural Network for the Visual Cortex (V1) Layer 5
============================

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
