import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib import animation
import opengen as og

from dynamics import BicycleModel
from track_generator import TrackGenerator
from reference_generator import ReferenceGenerator

mng = og.tcp.OptimizerTcpManager("optimizer/bicycle_model")
# Start the TCP server
mng.start()

# Simulation config
Ts = 0.01
N = 50
nz = 6
nu = 3
simulation_steps = 3000
bm = BicycleModel(Ts)
plot_car_trail = True
num_simultaneous_obstacles_to_track = 2

# Add road track
inner_a = 5
inner_b = 4
outer_a = 6
outer_b = 5
tg = TrackGenerator(inner_track_radii=[inner_a, inner_b], 
                    outer_track_radii=[outer_a, outer_b], 
                    num_sample_positions_centre_track_per_meter=40,
                      num_track_samples_per_meter=5)

# Generate reference points
rg = ReferenceGenerator(mpc_horizon=N, 
                        reference_track_positions_nby2=tg.track_centre_edge, 
                        refer_fixed_num_points_ahead=120,
                        num_points_obstacle_look_ahead=120,
                        advance_factor=1.5, states_string="x y psi vx vy omega")


# Add obstacles
centre_a = tg.centre_track_a
centre_b = tg.centre_track_b

inner_obstacle_lane_a = (inner_a + centre_a)/2
inner_obstacle_lane_b = (inner_b + centre_b)/2
outer_obstacle_lane_a = (outer_a + centre_a)/2
outer_obstacle_lane_b = (outer_b + centre_b)/2

inner_obstacle_lane_boundaries = tg.generate_elliptical_boundaries([inner_obstacle_lane_a, inner_obstacle_lane_b],
                                                                   [0, 0], 40, 0)[-1]
outer_obstacle_lane_boundaries = tg.generate_elliptical_boundaries([outer_obstacle_lane_a, outer_obstacle_lane_b],
                                                                   [0, 0], 40, 0)[-1]

inner_obstacles_ab = [(centre_a - inner_a)/2 + .1, (centre_b - inner_b)/2]
outer_obstacles_ab = [(outer_a - centre_a)/2 + .1, (outer_b - centre_b)/2]
centre_obstacles_ab = [(inner_obstacles_ab[0] + outer_obstacles_ab[0])/10 + .01, (inner_obstacles_ab[1] + outer_obstacles_ab[1])/10]
obstacles = {
                0: {"lane":"i", "idx":315, "xy":inner_obstacle_lane_boundaries[315], "ab":inner_obstacles_ab, "boundaries":None, "data":None, "idx_velocity":80},
                1: {"lane":"o", "idx":315, "xy":outer_obstacle_lane_boundaries[315], "ab":outer_obstacles_ab, "boundaries":None, "data":None, "idx_velocity":80},
                # 2: {"lane":"c", "idx":600, "xy":tg.track_centre_edge[600], "ab":centre_obstacles_ab, "boundaries":None, "data":None, "idx_velocity":10//Ts},
                3: {"lane":"i", "idx":945, "xy":inner_obstacle_lane_boundaries[945], "ab":inner_obstacles_ab, "boundaries":None, "data":None, "idx_velocity":80},
                4: {"lane":"o", "idx":945, "xy":outer_obstacle_lane_boundaries[945], "ab":outer_obstacles_ab, "boundaries":None, "data":None, "idx_velocity":80}
            }

for obstacle_id in obstacles:
    xy = obstacles[obstacle_id]["xy"]
    ab = obstacles[obstacle_id]["ab"]
    obstacle_boundaries, obstacle_data = tg.generate_obstacle_ellipse(obstacle_radii_ab=ab,
                                                   obstacle_pos_xy=xy,
                                                   num_obstacle_samples_per_meter=10,
                                                   rotate_major_axis_by_rad=tg.slope_ellipse_rad(xy=xy, ab=ab))
    obstacles[obstacle_id]["boundaries"] = obstacle_boundaries
    obstacles[obstacle_id]["data"] = obstacle_data
    rg.add_obstacle(obstacle_boundaries_nby2=obstacle_boundaries, obstacle_data=obstacle_data)

# Run simulations
current_position = rg.reference_track_positions_nby2[0]
z_state_0 = np.array([*current_position, np.pi/2, 0, 0, 0]).reshape(nz, 1)
pos_ref, is_obstacle_in_range, obstacles_in_range_ids = rg.get_new_reference_point(z_state_0.flatten(), enable_refer_fixed_num_points_ahead=True,
                                    motion_direction="ccw", ignore_path_refs_idx_to_avoid=True, return_obstacle_in_range_alert=True)

# initialise obstacles data
obstacles_data = [0]*num_simultaneous_obstacles_to_track
for obstacle_number in range(num_simultaneous_obstacles_to_track):
    obstacles_data[obstacle_number] = [0, 0, 1, 1, 0, 0]

# set obstacles and is obstacle present data at t=0 
for i, obstacle_id in enumerate(obstacles_in_range_ids):
    if i >= num_simultaneous_obstacles_to_track:
        break
    #                   [cx, cy, a, b, slope, is_obstacle_present]
    obstacles_data[i] = [0, 0, 1, 1, 0, 0]  # use 1 as radii to avoid div by zero
    boundaries, data = rg.obstacles[obstacle_id]
    obstacles_data[i][:2] = data["centre"]
    obstacles_data[i][2:4] = data["radii"]
    obstacles_data[i][2] += bm.car_width
    obstacles_data[i][3] += bm.car_width
    obstacles_data[i][4] = data["major_ax_rot_rad"]
    obstacles_data[i][5] = 1    

# initial state and input references
z_ref = np.array([*pos_ref, 0, 0, 0, 0]).reshape(nz, 1)
print(f"z_0: {z_state_0} \t z_ref: {z_ref}")
u_ref = np.array([0, 0, 0]).reshape(nu, 1)
u_prev = np.array([0, 0, 0]).reshape(nu, 1)

# simulation cache for plotting
state_sequence = np.zeros((simulation_steps, nz))
z_ref_seq = np.zeros([simulation_steps, nz])
z_ref_seq[0, :] = np.array([*pos_ref, 0, 0, 0, 0]).reshape(nz)
car_projection_seq = np.zeros([simulation_steps, nz])
car_current_projection_pos = rg.reference_track_positions_nby2[rg.current_projection_idx]
car_projection_seq[0, :] = np.array([*car_current_projection_pos, 0, 0, 0, 0]).reshape(nz)
obstacles_pos_seq = dict()
obstacles_pos_seq[0] = dict()
input_sequence = np.zeros((simulation_steps, nu))
state_sequence[0, :] = np.array(z_state_0.reshape(nz))
obstacles_appearence_sequence = dict()
# record obstacle appearance sequence for each obstacle for visualisation
# this will be used to turn obstacles visible and invisible on the track based
# in whether the obstacle is in the virtual sensor range or not respectively
for obstacle_id in rg.obstacles:
    obstacles_appearence_sequence[obstacle_id] = [False]
    if obstacle_id in obstacles_in_range_ids:
        obstacles_appearence_sequence[obstacle_id] = [True]
    obstacles_pos_seq[0][obstacle_id] = rg.obstacles[obstacle_id][0]

z_current = z_state_0
for k in range(simulation_steps-1):
    # pass NMPC problem parameters
    solver_response = mng.call(p=[
                                    *z_current.flatten(), *z_ref.flatten(), 
                                    *u_ref.flatten(), *u_prev.flatten(), 
                                    *[data for obstacle_data in obstacles_data for data in obstacle_data]  # unpack all obstacle's data in order
                                  ])
    assert solver_response.is_ok(), "solver failed!"
    out = solver_response.get()

    # get the set of solutions (N control actions)
    us = out.solution
    # print some useful info for tuning
    print(k, us[0:nu], z_current[0:5], z_ref.flatten()[:2], is_obstacle_in_range, len(obstacles_in_range_ids))

    # use the first control action from the computed sequence of solutions and simulate one time step
    u_mpc = us[0:nu]
    z_next = np.array(bm.dynamics_generic_dt(z_current, u_mpc, lib=np))
    z_next_r = np.reshape(np.array(z_next), (nz, 1))
    u_prev = np.array(u_mpc)
    # cache results
    state_sequence[k+1, :] = z_next.T
    input_sequence[k, :] = u_mpc
    z_current = z_next.flatten()

    # get new reference position and cache
    pos_ref, is_obstacle_in_range, obstacles_in_range_ids = rg.get_new_reference_point(z_current.flatten(), enable_refer_fixed_num_points_ahead=True,
                                        motion_direction="ccw", ignore_path_refs_idx_to_avoid=True, return_obstacle_in_range_alert=True)
    z_ref_seq[k+1, :] = np.array([*pos_ref, 0, 0, 0, 0]).reshape(nz)
    car_current_projection_pos = rg.reference_track_positions_nby2[rg.current_projection_idx]
    car_projection_seq[k+1, :] = np.array([*car_current_projection_pos, 0, 0, 0, 0]).reshape(nz)
    z_ref = np.array([*pos_ref, 0, 0, 0, 0]).reshape(nz, 1)

    # print detected obstacles and cache
    print([obstacle_data for obstacle_data in obstacles_data])
    for obstacle_id in rg.obstacles:
        if obstacle_id in obstacles_in_range_ids:
            obstacles_appearence_sequence[obstacle_id].append(True)
        else:
            obstacles_appearence_sequence[obstacle_id].append(False)
    
    # reset obstacle data for next iteration
    if is_obstacle_in_range:
        for i, obstacle_id in enumerate(obstacles_in_range_ids):
            if i >= num_simultaneous_obstacles_to_track:
                break
            obstacles_data[i] = [0, 0, 1, 1, 0, 0]
            boundaries, data = rg.obstacles[obstacle_id]
            obstacles_data[i][:2] = data["centre"]
            obstacles_data[i][2:4] = data["radii"]
            obstacles_data[i][2] += bm.car_width
            obstacles_data[i][3] += bm.car_width
            obstacles_data[i][4] = data["major_ax_rot_rad"]
            obstacles_data[i][5] = 1  
    else:
        for i, obstacle_id in enumerate(obstacles_data):
            obstacles_data[i] = [0, 0, 1, 1, 0, 0]
    
    # move obstacles by their defined velocity (approximately)
    xys = []
    for obstacle_id in obstacles:
        idx_velocity = obstacles[obstacle_id]["idx_velocity"]
        obstacles[obstacle_id]["idx"] += idx_velocity*Ts
        obstacles[obstacle_id]["idx"] = int(np.ceil(obstacles[obstacle_id]["idx"]))
        if obstacles[obstacle_id]["lane"].lower() == "i":
            obstacles[obstacle_id]["idx"] %= len(inner_obstacle_lane_boundaries)
            xy = inner_obstacle_lane_boundaries[obstacles[obstacle_id]["idx"]]
        elif obstacles[obstacle_id]["lane"].lower() == "o":
            obstacles[obstacle_id]["idx"] %= len(outer_obstacle_lane_boundaries)
            xy = outer_obstacle_lane_boundaries[obstacles[obstacle_id]["idx"]]
        elif obstacles[obstacle_id]["lane"].lower() == "c":
            obstacles[obstacle_id]["idx"] %= len(tg.track_centre_edge)
            xy = tg.track_centre_edge[obstacles[obstacle_id]["idx"]]
        else:
            print("obstacles[obstacle_id]['idx'] can only be 'i' inner, 'o' outer, and 'c' centre obstacle lanes.")
        xys.append(xy)
    rg.move_obstacles(xys, tg=tg)
    # cache obstacle positions
    obstacles_pos_seq[k+1] = dict()
    for id, obstacle_id in zip(obstacles, rg.obstacles):
        obstacles[id]["boundaries"] = rg.obstacles[obstacle_id][0]
        obstacles[id]["data"] = rg.obstacles[obstacle_id][1]
        obstacles_pos_seq[k+1][obstacle_id] = rg.obstacles[obstacle_id][0]

# kill the server once simulation in complete
mng.kill()

# setup for visualisation
car_width_half = bm.car_width / 2
car_length_half = bm.car_length / 2
car_roof_length_frac = .9
car_roof_width_frac = .95
psi = z_state_0[2]
x = z_state_0[0] - (np.cos(psi) * car_length_half - np.sin(psi) * car_width_half)
y = z_state_0[1] - (np.sin(psi) * car_length_half + np.cos(psi) * car_width_half)
car_body = Rectangle((x, y), bm.car_length, bm.car_width, fc="red")
car_roof = Rectangle((x + car_length_half, y + bm.car_width * (1 - car_roof_width_frac)), car_length_half * car_roof_length_frac, bm.car_width * car_roof_width_frac, fc="yellow")
car_ref_point = Circle((z_ref_seq[0, 0], z_ref_seq[0, 1]), 0.1, fc="red")
car_projection_point = Circle((car_current_projection_pos[0], car_current_projection_pos[1]), 0.1, fc="white")

# setup plot for visualising simulation results
fig = plt.figure(figsize=(16, 9))
grid = fig.add_gridspec(9, 16)
time = np.linspace(0, Ts * simulation_steps, simulation_steps)

pos_ax = fig.add_subplot(grid[0:6, 0:6])
car_trail, = pos_ax.plot([], [])
tg.add_track_patch_to_plt_axs(pos_ax)
pos_ax.set_aspect('equal')
pos_ax.grid()
pos_ax.minorticks_on()
pos_ax.set_xlabel("x-position (m)")
pos_ax.set_ylabel("y-position (m)")
vel_ax = fig.add_subplot(grid[0:2, 7:16])
vel_ax.grid()
vel_ax.set_xlim(time[0], time[-1])
vel_ax.set_ylim(np.min(state_sequence[:, 3:5]) - 0.2, np.max(state_sequence[:, 3:5]) + 0.2)

acc_brk_ax = fig.add_subplot(grid[3:6, 7:16])
acc_brk_ax.set_xlim(time[0], time[-1])
acc_brk_ax.grid()
acc_brk_ax.set_ylim(0 - 0.1, 1.1 + 0.1)

omega_ax = fig.add_subplot(grid[7:9, 0:7])
omega_ax.grid()
omega_ax.set_xlim(time[0], time[-1])
omega_ax.set_ylim(np.min(state_sequence[:, 5]) - 0.2, np.max(state_sequence[:, 5]) + 0.2)

delta_ax = fig.add_subplot(grid[7:9, 8:16])
delta_ax.grid()
delta_ax.set_xlim(time[0], time[-1])
delta_ax.set_ylim(np.min(input_sequence[:, 1]) - 0.2, np.max(input_sequence[:, 1]) + 0.2)


x_lims = [-outer_a, outer_a]
y_lims = [-outer_b, outer_b]

# add vehicle graphics to plot
pos_ax.add_patch(car_body)
pos_ax.add_patch(car_roof)
pos_ax.add_patch(car_ref_point)
pos_ax.add_patch(car_projection_point)

# display obstacle safety circles as dotted boundaries
obstacle_scatters = []
for obstacle in rg.obstacles:
    scatter_plt = pos_ax.scatter(rg.obstacles[obstacle][0][:, 0], rg.obstacles[obstacle][0][:, 1], color="cyan", s=3)
    obstacle_scatters.append(scatter_plt)
pos_ax.set_xlim(min(x_lims) - 1, max(x_lims) + 1)
pos_ax.set_ylim(min(y_lims) - 1, max(y_lims) + 1)

# setup empty axes for animation
vx_line, = vel_ax.plot([], [])
vy_line, = vel_ax.plot([], [])
vel_ax.legend(["$v_x(t)$", "$v_y(t)$"])
vel_ax.set_xlabel("time (s)")
vel_ax.set_ylabel("Local velocity (m/s)")
omega_line, = omega_ax.plot([], [])
omega_ax.legend(["$\omega(t)$"])
omega_ax.set_xlabel("time (s)")
omega_ax.set_ylabel("Omega (rad/s)")

acc_line, = acc_brk_ax.plot([], [])
delta_line, = delta_ax.plot([], [])
delta_ax.legend(["$\delta(t)$"])
delta_ax.set_xlabel("time (s)")
delta_ax.set_ylabel("Delta (rad)")
brake_line, = acc_brk_ax.plot([], [])
acc_brk_ax.legend(["PWM(t)", "Brake"])
acc_brk_ax.set_xlabel("time (s)")
acc_brk_ax.set_ylabel("Controls (normalised)")

# initial frames at t=0
def init_frame():
    return car_body, car_roof, vx_line, vy_line, omega_line, acc_line, delta_line, brake_line

def init_frame_with_ref():
    return list(init_frame()) + [car_ref_point, car_projection_point] + ([car_trail] if plot_car_trail else [])

# functions to update animation frame
def update_frame(frame):
    x = state_sequence[frame, 0]
    y = state_sequence[frame, 1]
    psi = state_sequence[frame, 2]

    t = time[0:frame]
    vx = state_sequence[0:frame, 3]
    vy = state_sequence[0:frame, 4]
    omega = state_sequence[0:frame, 5]

    acc = input_sequence[0:frame, 0]
    delta = input_sequence[0:frame, 1]
    brake = input_sequence[0:frame, 2]

    x -= np.cos(psi) * car_length_half - np.sin(psi) * car_width_half
    y -= np.sin(psi) * car_length_half + np.cos(psi) * car_width_half
    tx = np.cos(psi) * car_length_half - np.sin(psi) * (bm.car_width * (1 - car_roof_width_frac))
    ty = np.sin(psi) * car_length_half + np.cos(psi) * (bm.car_width * (1 - car_roof_width_frac))
    
    car_body.set_xy((x, y))
    car_body.angle = psi * 180 / np.pi
    car_roof.set_xy((x + tx, y + ty))
    car_roof.angle = psi * 180 / np.pi

    vx_line.set_data(t, vx)
    vy_line.set_data(t, vy)
    omega_line.set_data(t, omega)
    acc_line.set_data(t, acc)
    delta_line.set_data(t, delta)
    brake_line.set_data(t, brake)

    return car_body, car_roof, vx_line, vy_line, omega_line, acc_line, delta_line, brake_line

def update_frame_with_ref(frame):
    x_ref = z_ref_seq[frame, 0]
    y_ref = z_ref_seq[frame, 1]

    x = state_sequence[0:frame, 0]
    y = state_sequence[0:frame, 1]

    x_proj = car_projection_seq[frame, 0]
    y_proj = car_projection_seq[frame, 1]

    car_ref_point.set_center((x_ref, y_ref))
    car_projection_point.set_center((x_proj, y_proj))
    car_trail.set_data(x, y)
    
    for i, obstacle_id in enumerate(obstacles_appearence_sequence):
        if obstacles_appearence_sequence[obstacle_id][frame] == True:
            obstacle_scatters[i].set_offsets(np.c_[obstacles_pos_seq[frame][obstacle_id][:, 0], obstacles_pos_seq[frame][obstacle_id][:, 1]])
            obstacle_scatters[i].set_visible(True)
        else:        
            obstacle_scatters[i].set_visible(False)   

    return list(update_frame(frame=frame)) + [car_ref_point, car_projection_point] + ([car_trail] if plot_car_trail else [])

# animate and save the simulation results
car_animation = animation.FuncAnimation(fig, update_frame_with_ref, frames=state_sequence.shape[0], init_func=init_frame_with_ref, interval=int(Ts*1000), blit=True)
car_animation.save('NMPC_moving_obstacles2.mp4', writer='ffmpeg', fps=30, bitrate=1800, dpi=400)
plt.show()  # display the animation