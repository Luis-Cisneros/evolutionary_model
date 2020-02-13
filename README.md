## Evolutionary Model for Blockchain Mining Pool Selection

Evolutionary model using pairwise proportional imitation built in accordance to the paper "Evolutionary Game for Mining Pool Selection in Blockchain Networks"  by Xiaojun Liu, et al.

The model built in **evolutionary_model.py** must run for a large enough time and with enough miner *n* so that it leads asymptotically to the following system of ordinary differential equations:

*x'<sub>i</sub>(t) = x<sub>i</sub>(t)(y<sub>i</sub>(x(t); w) - y̅<sub>i</sub>(x(t)))*

For all *i* in *{1, ..., M}* where *M* is the number of mining pools available.

Where:

*x(t)* is the population state at time *t*.

*w* is the strategy profile measured in PentaHashes per cycle.

*y<sub>i</sub>(x(t); w)* is the expected payoff of the ith group given the strategy profile *w*.

*y̅<sub>i</sub>(x(t))* is the average expected payoff of all groups.

The population state at end of the model (*x\**) is tested for Nash Equilibrium and Evolutionary Stable State criteria in **stable_strategy.py**

*x\** is a Nash Equilibrium of the model if: *(x - x\*) Y(x\*) ≤ 0*  for all possible population states x


*x\** is an Evolutionary Stable State of the model if: *(x\* - x) Y(x) ≥ 0*  for all population states x that follow the condition: *(x - x\*) Y(x\*) = 0* 

The executable **main.py** must receive the necessary network and miner parameters.

Code optimized using the nijt compiler from numba.
