## Brief
An NMPC controlled small scale autonomous car driving around a road track with moving obstacles. The vehicle dynamics and reference generator were inspired and built upon details provided in [[1]](#references). 

An explanation of the theory and implementation of the simulation code is available in [NMPC part 1](https://am-press.github.io/posts/maths/nmpc-two-wheel-bicycle-model/) and [NMPC part 2 (coming soon)]().
If you like the posts and want to build your own NMPC formulations, please check out [OpEn](https://alphaville.github.io/optimization-engine/). I thank my guide [Pantelis Sopasakis](https://github.com/alphaville) for his support in writing the articles and developing this code.

## Usage
1. Run the file `build_optimiser.py` to build the NMPC optimiser formulation.
2. Run the file `simulation.py` to simulate and visualise the NMPC controlled small scale car.

## Results
The following GIFs showcase some of the simulation results presented in the [NMPC part 2](https://am-press.github.io/posts/maths/nmpc-two-wheel-bicycle-model/) article.

<div>
<img src="https://github.com/patchedByBatman/Autonomous_car_NMPC/blob/main/results/mpc_car_obst_2m_6.gif" alt="Simulation results: NMPC controlled car navigating around a big blue circle (obstacle) of radius 2m at the origin. The vehicle starts at (-2,-2) and reaches (1, 6)." width="640" height="400">
</div>
The above GIF shows how the NMPC controller navigates the vehicle around an obstacle of $2$m radius (represented by blue circle) situated at the origin. The vehicle starts at $(-2, -2)$ and successfully reaches its destination $(1, 6)$ without hitting the obstacle.


The NMPC controller is then commanded to keep moving on an elliptical track in counter-clockwise direction, without exiting the track. The above GIF showcases the corresponding results.
<div>
<img src="https://github.com/patchedByBatman/Autonomous_car_NMPC/blob/main/results/track_drive_higher_resolution_track.gif" alt="Simulation results: NMPC controlled car navigating on a track, sampled at 40 samples/m and 120 samples look ahead." width="640" height="400">
</div>

We then add four static obstacles, represented with blue elliptical dotted regions, to the track and command the NMPC controller to navigate the vehicle on the track without hitting the obstacles. The below GIF shows the corresponding simulation results.
<div>
<img src="https://github.com/patchedByBatman/Autonomous_car_NMPC/blob/main/results/track_drive_multiple_obs_no_disappear.gif" alt="Simulation results: NMPC controlled car navigating on a track, sampled at 40 samples/m and 120 samples look ahead." width="640" height="400">
</div>
As it can be see from the above GIF, the vehicle successfully navigates around the track without hitting any obstacles.

To add realism to the simulation, we introduce the concept of a virtual sensor, due to which the obstacles only appear when they are within the range of the virtual sensor. The range of the virtual sensor starts from the point of center-of-gravity of the car till the red dot shown in the figure.
<div>
<img src="https://github.com/patchedByBatman/Autonomous_car_NMPC/blob/main/results/track_drive_multiple_obs_disappear.gif" alt="Simulation results: NMPC controlled car navigating on a track, sampled at 40 samples/m and 120 samples look ahead." width="640" height="400">
</div>

Finally, we introduce moving obstacles to the track and see how the NMPC controller performs. The below GIF shows the corresponding simulation results.
<div>
<img src="https://github.com/patchedByBatman/Autonomous_car_NMPC/blob/main/results/NMPC_moving_obstacles.gif" alt="Simulation results: NMPC controlled car navigating on a track, sampled at 40 samples/m and 120 samples look ahead." width="640" height="400">
</div>
It is evident, from the above GIF, that the NMPC controller is successful in navigating the car around the track whilst avoiding multiple moving obstacles.

## File brief
1. `NMPC_moving_obstacles.gif` shows a sample simulation result.
2. `build_optimiser.py` contains code to build an NMPC optimiser using `OpEn`.
3. `dynamics.py` contains the bicycle model dynamics of a small scale car [[1]](#references).
4.  `track_generator.py` contains code to generate a road track and obstacle safety circles.
5. `reference_generator.py` contains code for generating position references for the NMPC controller and do **DOSA**.
6. `simulation.py` contains code to simulate the NMPC controlled vehicle driving around a track with moving obstacles.

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
