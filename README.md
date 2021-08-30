# paper_rl_thermal_machines
Code used to produce the results presented in the manuscript "*Identifying optimal cycles in quantum thermal machines with reinforcement-learning*" by P.A. Erdman and F. Noé.

## Getting started
To get started, open the [```jupyter```](jupyter) folder. The following notebooks allow you to train an RL agent to control the corresponding quantum thermal machine:
* [```two_level_heater.ipynb```](jupyter/two_level_heater.ipynb)
* [```two_level_engine.ipynb```](jupyter/two_level_engine.ipynb)
* [```superconducting_qubit_refrigerator.ipynb```](jupyter/superconducting_qubit_refrigerator.ipynb)
* [```harmonic_engine.ipynb```](jupyter/harmonic_engine.ipynb)

The values of the parameters in these notebooks correspond to the ones used in the manuscript.

The following notebooks allow you to produce plots in the style of the figures of the manuscript using training data produced by the previous notebooks:
* [```paper_main_plots.ipynb```](jupyter/paper_main_plots.ipynb)
* [```superconducting_qubit_refrigerator.ipynb```](jupyter/superconducting_qubit_refrigerator.ipynb)

## Acknowledgement
Implementation of the soft actor-critic method based on modifications of the code provided at:

J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).
