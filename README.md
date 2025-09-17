## Brief
An NMPC controlled small scale autonomous car driving around a road track with moving obstacles. The vehicle dynamics and reference generator were inspired and built upon details provided in [[1]](#references). 

An explanation of the theory and implementation of the simulation code is available in [NMPC part 1](https://am-press.github.io/posts/maths/nmpc-two-wheel-bicycle-model/) and [NMPC part 2](https://am-press.github.io/posts/maths/nmpc-two-wheel-bicycle-model/).
If you like the posts and want to build your own NMPC formulations, please check out [OpEn](https://alphaville.github.io/optimization-engine/). I thank my guide [Pantelis Sopasakis](https://github.com/alphaville) for his support in writing the articles and developing this code.

## Usage
1. Run the file `build_optimiser.py` to build the NMPC optimiser formulation.
2. Run the file `simulation.py` to simulate and visualise the NMPC controlled small scale car.

## File brief
1. `NMPC_moving_obstacles.gif` shows a sample simulation result.
2. `build_optimiser.py` contains code to build an NMPC optimiser using `OpEn`.
3. `dynamics.py` contains the bicycle model dynamics of a small scale car [[1]](#references).
4.  `track_generator.py` contains code to generate a road track and obstacle safety circles.
5. `reference_generator.py` contains code for generating position references for the NMPC controller and do **DOSA**.
6. `simulation.py` contains code to simulate the NMPC controlled vechile driving around a track with moving obstacles.

## Dependencies
The software used for the simulations are:
1. [Python3](https://www.python.org/): An open-source high-level interpreted programming language.
2. [NumPy](https://numpy.org/): Numerical Python in short NumPy is an open-source scientific computational tool for multidimensional array operations. 
3. [Matplotlib](https://matplotlib.org/): Is an open-source visualisation library for Python.
4. [CasADi](https://web.casadi.org/): Is an open-source tool for nonlinear optimisation and algorithmic (automatic) differentiation. 
5. [Python-control](https://python-control.readthedocs.io/en/0.10.1/): An open-source control systems library for Python.
6. [OpEn](https://alphaville.github.io/optimization-engine/): Optimization Engine in full, is a framework for designing and solving non-linear and non-convex optimisation problems. 


## References
[1] Cataffo, Vittorio & Silano, Giuseppe & Iannelli, Luigi & Puig, Vicenç & Glielmo, Luigi. (2022). A Nonlinear Model Predictive Control Strategy for Autonomous Racing of Scale Vehicles. 10.1109/SMC53654.2022.9945279.
