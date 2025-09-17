import numpy as np
from matplotlib import pyplot as plt
import opengen as og
import casadi.casadi as cs
import control as ctrl
from dynamics import BicycleModel
from functools import partial

# NMPC parameter configuration
nz = 6  # number of states
nu = 3  # number of inputs
N = 50  # prediction horizon
num_data_params_per_obstacle = 6  # assuming elliptical obstacles: [centre(x, y), radii(a, b), theta, 1 if obstacle_present else 0]
num_simultaneous_obstacles_to_track = 2 

# Bicycle model dynamics
Ts = 0.01  # sampling time
bm = BicycleModel(Ts)

# State and input boundaries
zmin = [-100, -100, -np.inf, 0, 0, -np.inf]
zmax = [100, 100, np.inf, 5, 5, np.inf]
umin_seq = [0, -np.pi/3, 0] * N
umax_seq = [1, np.pi/3, 1] * N

# NMPC penalty weights
Q = np.diagflat([1000, 1000, 0, 0, 0, 0])
R = np.diagflat([600, 2, 0.1])
R2 = 100

# NMPC formulation state and input parameter sequences
u_seq = cs.SX.sym("u_seq", N * nu, 1)
problem_params = cs.SX.sym("problem_params", 2*nz + 2*nu + 
                           num_simultaneous_obstacles_to_track * num_data_params_per_obstacle, 1) 
                         # [z_0, z_ref, u_ref, u_{-1}, n*[obstacle_centre(x, y), obstacle_radii(a, b), rot_angle_rad, 1 if obstacle_present else 0]

# NMPC initial settings
z_0 = problem_params[ : nz]  # initial state
z_ref = problem_params[nz : 2 * nz]  # reference destination
u_ref = problem_params[2 * nz : 2 * nz + nu]  # reference inputs, currently un-used
u_prev = problem_params[2 * nz + nu : 2 * nz + 2 * nu]  # u_{-1} value of inputs before t=0

# implementation of special function for DOSA
obstacles_functions = []  # list of obstacle function instances for tracking multiple obstacles
obstacle_params = [0]*num_simultaneous_obstacles_to_track  # list of lists of obstacle params, contains one list per obstacle

def obstacle_function(x, y, obstacle_param):
    """Computes and returns obstacle avoidance constrain for one instant.
    
    :param x: current x position of the vehicle.
    :param y: current y position of the vehicle.
    :param obstacle_param: a length 6 list of obstacle params of structure 
                            [centre(x, y), radii(a, b), theta, 1 if obstacle_present else 0]
    :return: obstacle avoidance constraint for one instant.
    """
    # with obstacle at centre
    X_ = (x - obstacle_param[0])
    Y_ = (y - obstacle_param[1])
    # obstacle's major and minor axis radii
    a = obstacle_param[2]
    b = obstacle_param[3]
    # obstacle's major axis slope wrt global x-axis
    theta = obstacle_param[4]
    # obstacle presence flag, 1 if an obstacle is in virtual sensor's range, else 0
    is_present = obstacle_param[5]

    # compute the elliptical obstacle avoidance constraint using translation and rotation of axis.
    # set the ellipse's centre at the obstacle's centre and rotated to match the obstacles slope.
    X = (X_ * cs.cos(theta) + Y_ * cs.sin(theta)) / (a + bm.car_width/2)
    Y = (-X_ * cs.sin(theta) + Y_ * cs.cos(theta)) / (b + bm.car_width/2)

    return cs.fmax(0, is_present - X ** 2 - Y ** 2)

# loop to create instances of obstacle function. One function instance per expected object to be tracked.
for i in range(num_simultaneous_obstacles_to_track):
    obstacle_params[i] = problem_params[2*(nz + nu) + num_data_params_per_obstacle*i:2*(nz + nu) + num_data_params_per_obstacle*(i+1)]
    obstacles_functions.append(partial(obstacle_function, obstacle_param=obstacle_params[i]))

# define the stage cost
def stage_cost(u_current, u_prev):
    return (u_current.T - u_prev.T) @ R @ (u_current - u_prev) + R2 * u_current[0] * u_current[2]

# define the terminal cost
def terminal_cost(z):
    return (z.T - z_ref.T) @ Q @ (z - z_ref)

# initialise constraints
z_t = z_0
total_cost = 0
penalty_constraint_obstacles = 0
penalty_constraint_tracks = 0
penalty_constraint_vx = 0

# formulate the NMPC total cost and constraints
for t in range(N):
    u_current = u_seq[t * nu : (t + 1) * nu]  # set current time step input
    total_cost += stage_cost(u_current, u_prev)  # add stage cost to total cost for the current time step
    z_t = bm.dynamics_cs(z_t, u_current)  # set current state to resultant state of the current input
    u_prev = u_current  # change previous input to current input for next iteration
    penalty_constraint_vx += cs.fmax(0, cs.norm_1(z_t[3]) - zmax[3])  # add set vx maximum constraint
    penalty_constraint_vx += cs.norm_1(cs.fmin(zmin[3], z_t[3]))  # add set vx minimum constraint
    # add obstacle constraints
    for obstacle_function in obstacles_functions:
        penalty_constraint_obstacles += obstacle_function(z_t[0], z_t[1])
    # add penalties for exiting the track
    penalty_constraint_tracks += cs.fmax(0, 1 - (z_t[0]/(5 + bm.car_width/2))**2 - (z_t[1]/(4 + bm.car_width/2))**2)
    penalty_constraint_tracks += cs.fmax(0, (z_t[0]/(6 - bm.car_width/2))**2 + (z_t[1]/(5 - bm.car_width/2))**2 - 1)

# add terminal cost to total cost
total_cost += terminal_cost(z_t)

# define rectangular constraints for inputs
U = og.constraints.Rectangle(umin_seq, umax_seq)

# define and build the problem
problem = og.builder.Problem(u_seq, problem_params, total_cost)
problem = problem.with_constraints(U)
problem = problem.with_penalty_constraints(penalty_constraint_vx 
                                           + penalty_constraint_obstacles 
                                           + penalty_constraint_tracks)

build_config = og.config.BuildConfiguration() \
    .with_build_directory("optimizer") \
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta().with_optimizer_name("bicycle_model")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_penalty_weight_update_factor(3)

builder = []
builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config).with_verbosity_level(0)

builder.build()