# Hybrid Control
Code for planning in hybrid models on classic control suite.

[link to paper](https://openreview.net/forum?id=956TTbUHt8)

## Recurrent switching linear dynamical systems 
We exploit recurrent switching linear dynamical systems allowing us to seperate control into high level planning through discrete variables and low level *reflexive* LQR control. 

<img src="./figs/piecewise.png" width="400"/>

## Exploration

Due to the small dimension of the discrete variables, which only capture non-linear behaviour, we can calculate information theoretic bonuses for exploration.

<img src="./figs/exploration.png" width="400"/>


## Performance

The directed exploration allows the algorithm to solve sparse mountain car efficiently.

<img src="./figs/average_reward_plot.png" width="400"/>

<!-- ![](/figs/piecewise.png)
![](/figs/exploration.png)
![](/figs/average_reward_plot.png) -->
<!-- ![](/figs/coverage_HHA_IG.png) -->

## Installation
To setup use 
`pip install -e .` in the top level directory (where `setup.py` lives)
this will install `hybrid_control` as an editable package, which avoids the mess 
of relative imports and python paths.