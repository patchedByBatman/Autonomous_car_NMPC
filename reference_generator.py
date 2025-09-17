import numpy as np
from track_generator import TrackGenerator

class ReferenceGenerator:
    """Implements reference generator algorithm for the NMPC formulation.
    Provides methods to:
        1. Add obstacles to the track.
        2. Move obstacles around the track.
        3. Virtual sensor based obstacle detection.
    """
    def __init__(self, mpc_horizon, reference_track_positions_nby2, 
                 refer_fixed_num_points_ahead=10, num_points_obstacle_look_ahead=20, 
                 advance_factor=0.1, states_string="x vx y vy"):
        """Constructor.
        
        :param mpc_horizon: NMPC prediction horizon N.
        :param reference_track_positions_nby2: a n-by-2 numpy array of the sampled curve on 
                        which reference has to be generated. Example [[x1, y1], [x2, y2], ...]. 
        :param refer_fixed_num_points_ahead: No. of points ahead of the current projected vehicle's position, 
                        at which reference needs to be generated on the sampled curve. Default 10 samples.
        :param num_points_obstacle_look_ahead: No. of points ahead of the current projected vehicle's position,
                        till which obstacles are scanned for. Default 20 samples.
        :param states_string: A string representing the sequence in which the vehicle states [x, vx, y, vy] are stored.
                        Default "x vx y vy".
        """
        self.N = mpc_horizon
        self.reference_track_positions_nby2 = reference_track_positions_nby2
        self.num_ref_track_samples = reference_track_positions_nby2.shape[0]
        self.refer_fixed_num_points_ahead = refer_fixed_num_points_ahead
        self.num_points_obstacle_look_ahead = num_points_obstacle_look_ahead
        self.advance_factor = advance_factor
        
        self.obstacles = dict()
        self.last_obstacle_id = 0
        self.obstacle_projections_on_path = dict()
        self.path_refs_idx_to_avoid = set()
        self.current_projection_idx = 0


        self.xidx = None
        self.vxidx = None
        self.yidx = None
        self.vyidx = None
        self.update_state_idxs(states_string=states_string)

    def update_state_idxs(self, states_string="x vx y vy"):
        """Updates the state sequence of the vehicle.
        
        :param states_string: A string representing the sequence in which the vehicle states [x, vx, y, vy] are stored.
                        Default "x vx y vy".
        """
        states_list = states_string.strip().split()
        for idx, state in enumerate(states_list):
            if state.lower() == "x":
                self.xidx = idx
            elif state.lower() == "vx":
                self.vxidx = idx
            elif state.lower() == "y":
                self.yidx = idx
            elif state.lower() == "vy":
                self.vyidx = idx

    def euclidean_distance_squared(self, p1_1by2, p2_1by2):
        """Computes and returns the squared Euclidean distance between two given points.
        
        :param p1_1by2: point-1, numpy array of 1-by-2 dims. Example [x1, y1].
        :param p2_1by2: point-2, numpy array of 1-by-2 dims. Example [x2, y2].
        :return: Squared Euclidean distance between point-1 and point-2.
        """
        return np.sum((p1_1by2 - p2_1by2)**2)

    def find_projection(self, of_object_nby2, on_object_nby2):
        """Computes and returns the projected points of an object on a different object.
        
        :param of_object_nby2: Boundary coordinates of the object whose projection needs to be computed.
                                numpy array of n-by-2 dims. Example [[x1, y1], [x2, y2], ...].
        :param on_object_nby2:Boundary coordinates of the object on which the projection needs to be computed.
                                numpy array of n-by-2 dims. Example [[x1, y1], [x2, y2], ...].
        :return idx_proj_on_object: a set of indices of on_object_nby2, corresponding to the projection of of_object_nby2.
        """
        idx_proj_on_object = set()
        for a in of_object_nby2:
            min_dist = np.inf
            min_dist_idx = None
            for idx, b in enumerate(on_object_nby2):
                dist = self.euclidean_distance_squared(a, b)
                if dist <= min_dist:
                    min_dist = dist
                    min_dist_idx = idx
            idx_proj_on_object |= {min_dist_idx}
        return idx_proj_on_object

    def add_obstacle(self, obstacle_boundaries_nby2, obstacle_data):
        """Adds an obstacle for simulation.
        
        :param obstacle_boundaries_nby2: Boundary coordinates of the obstacle.
                                numpy array of n-by-2 dims. Example [[x1, y1], [x2, y2], ...]. 
        :param obstacle_data: obstacle data is a dict of form
                    {
                        "radii": [a, b], 
                        "centre": [cx, cy],
                        "num_samples": param num_obstacle_samples_per_meter,
                        "major_ax_rot_rad" = param rotate_major_axis_by_rad,
                        "circumference_approx" = approx circumference of the safety circle,
                        "num_sample_angles" = no. of samples,
                        "sample_angles" = sampled angles in range [0, 2pi],
                        "boundaries" = combined list of x and y boundary coordinates                        
                    }
        """
        self.obstacles[self.last_obstacle_id] = [obstacle_boundaries_nby2, obstacle_data]
        obstacle_projection_on_path = self.find_projection(obstacle_boundaries_nby2, self.reference_track_positions_nby2)
        self.obstacle_projections_on_path[self.last_obstacle_id] = list(obstacle_projection_on_path)
        self.path_refs_idx_to_avoid |= obstacle_projection_on_path
        self.last_obstacle_id += 1
    
    def move_obstacles(self, xys, tg:TrackGenerator):
        """Moves existing obstacles to positions with new centres provided.
        
        :param xys: new centres to which the existing obstacles need to be moved to.
                    list of dims n-by-2. Example [[cx1, cy1], [cx2, cy2], ....].
        :param tg: an object reference to TrackGenerator class instance used for simulation.
        """
        # soft warn if provided no. of new centres are unequal to the no. of existing obstacles. 
        if len(self.obstacles) != len(xys):
            print("warning reference_generator move_obstacles: len(self.obstacles) != len(xys). " \
                    "\nxys argument should be list of lists of dims nx2, where n is num obstacles.")
        
        # iterate over obstacles
        for obstacle_id, xy in zip(self.obstacles, xys):
            # copy old position data
            _, obstacle_data = self.obstacles[obstacle_id]
            obstacle_projection_on_path = set(self.obstacle_projections_on_path[obstacle_id])
            self.path_refs_idx_to_avoid -= obstacle_projection_on_path  # remove reference avoidance to old projection
            ab = obstacle_data["radii"]
            num_obstacle_samples_per_meter = obstacle_data["num_samples"]

            # generate a new safety circle with new centre and old dimensions
            obstacle_boundaries_nby2, obstacle_data = tg.generate_obstacle_ellipse(obstacle_radii_ab=ab,
                                                   obstacle_pos_xy=xy,
                                                   num_obstacle_samples_per_meter=num_obstacle_samples_per_meter,
                                                   rotate_major_axis_by_rad=tg.slope_ellipse_rad(xy=xy, ab=ab))
            
            # update with new position
            self.obstacles[obstacle_id] = [obstacle_boundaries_nby2, obstacle_data]
            obstacle_projection_on_path = self.find_projection(obstacle_boundaries_nby2, self.reference_track_positions_nby2)
            self.obstacle_projections_on_path[obstacle_id] = list(obstacle_projection_on_path)
            self.path_refs_idx_to_avoid |= obstacle_projection_on_path

    def is_obstacle_in_range(self, current_pos_idx):
        """Check if any obstacle is in range of the virtual sensor and return the corresponding data.
        
        :param current_pos_idx: current projected position of the obstacle centre on the reference curve (path).
        :return tuple(is_obstacle_in_range, obstacle_ids): 
                            flag is_obstacle_in_range is true when at least one obstacle is in virtual sensor's range. 
                            list obstacle_ids contains the list of obstacle ids in range. 
        """
        look_up_idx = current_pos_idx + self.num_points_obstacle_look_ahead
        is_obstacle_in_range = False
        obstacle_ids = []
        # iterate over all existing obstacles
        for obstacle_id in self.obstacle_projections_on_path:
            idxs = self.obstacle_projections_on_path[obstacle_id]  # get the obstacle's projection (indices) on the reference path
            # iterate over the obstacle projections
            for idx in idxs:
                # detect the obstacle any of its projections fall within the virtual sensor range
                if current_pos_idx <= idx <= look_up_idx:
                    is_obstacle_in_range = True
                    obstacle_ids.append(obstacle_id)
                    break
                # if obstacle crosses zeroth position but vehicle is still behind 
                # i.e., obstacle crossed the starting point before and right in front of the vehicle
                elif current_pos_idx <= self.num_ref_track_samples <= look_up_idx :
                    if 0 <= idx <= look_up_idx%self.num_ref_track_samples:
                        is_obstacle_in_range = True
                        obstacle_ids.append(obstacle_id)
                        break
        return is_obstacle_in_range, obstacle_ids

    def get_new_reference_point_idx(self, current_state_flatened, enable_refer_fixed_num_points_ahead=False,
                                     motion_direction="ccw", ignore_path_refs_idx_to_avoid=False, 
                                     return_obstacle_in_range_alert=False):
        """Compute and return a new reference point index for the NMPC.
        
        :param current_state_flatened: current state of the vehicle dynamics.
        :param enable_refer_fixed_num_points_ahead: Boolean to indicate whether to compute a reference 
                            that is fixed number of points ahead of the vehicle's current position. Default False.
        :param motion_direction: direction of circular motion of the vehicle. Default "ccw".
        :param ignore_path_refs_idx_to_avoid: Boolean, whether to ignore generate references close to obstacles.
                                                True means references are allowed to be close to obstacles. Default False.
        :param return_obstacle_in_range_alert: Boolean, whether to return an alert if an obstacle is detected. Default False.
        :return tuple(new_ref_idx, is_obstacle_in_range, obstacle_ids) or new_ref_idx: Self-explanatory.
        """
        # current state of the vehicle
        x = current_state_flatened[self.xidx]
        vx = current_state_flatened[self.vxidx]
        y = current_state_flatened[self.yidx]
        vy = current_state_flatened[self.vyidx]
        current_pos = np.array([[x, y]])
        current_pos_idx = self.find_projection(current_pos, self.reference_track_positions_nby2)
        current_pos_idx = list(current_pos_idx)[0]  #  take the front most edge of the vehicle as its current position.
        self.current_projection_idx = current_pos_idx
        is_obstacle_in_range, obstacle_ids = self.is_obstacle_in_range(current_pos_idx=current_pos_idx)

        if enable_refer_fixed_num_points_ahead:
            if motion_direction.lower() == "ccw":
                new_ref_idx = (current_pos_idx + self.refer_fixed_num_points_ahead) % self.num_ref_track_samples
            elif motion_direction.lower() == "ccw":
                new_ref_idx = (current_pos_idx - self.refer_fixed_num_points_ahead) % self.num_ref_track_samples
            else:
                Warning(f"motion direction can be 'ccw' for counter clock-wise rotation or 'cw' for clock-wise, not {motion_direction}." 
                        "\nAssuming default 'ccw' and continuing...")
                new_ref_idx = (current_pos_idx + self.refer_fixed_num_points_ahead) % self.num_ref_track_samples
            if return_obstacle_in_range_alert:
                # set the new reference in-between the first two obstacles if there are 2 or more obstacles in range.
                if len(obstacle_ids) > 1:
                    obs0 = min(self.obstacle_projections_on_path[obstacle_ids[0]])
                    obs1 = max(self.obstacle_projections_on_path[obstacle_ids[1]])
                    new_ref_idx = (obs0 + obs1)//2
                    if motion_direction.lower() == "ccw":
                        # when moving in ccw, if the new reference falls behind the vehicle, recompute.
                        # this happens when one of the obstacles cross the starting point before the other obstacle and the vehicle.
                        if new_ref_idx <= current_pos_idx:
                            obs0 = max(self.obstacle_projections_on_path[obstacle_ids[0]])
                            new_ref_idx = max((obs0 + obs1 + 10)%self.num_ref_track_samples, (obs0 + obs1)//2)
                    elif motion_direction.lower() == "cw":
                        # when moving in cw, if the new reference falls behind the vehicle, recompute.
                        # this happens when one of the obstacles cross the starting point before the other obstacle and the vehicle.
                        if new_ref_idx >= current_pos_idx:
                            obs0 = max(self.obstacle_projections_on_path[obstacle_ids[0]])
                            new_ref_idx = min((obs0 + obs1 - 10)%self.num_ref_track_samples, (obs0 + obs1)//2)
                return new_ref_idx, is_obstacle_in_range, obstacle_ids
            return new_ref_idx

        # when fixed number of points ahead reference generation is disable
        # compute the new reference based on the current vehicle velocity and advance factor
        max_dist_x = x + vx*self.N*self.Ts*self.advance_factor
        max_dist_y = y + vy*self.N*self.Ts*self.advance_factor
        max_ref_pos = np.array([[max_dist_x, max_dist_y]])
        new_ref_idx = self.find_projection(max_ref_pos, self.reference_track_positions_nby2)
        new_ref_idx = list(new_ref_idx)[0]
        # if new reference is not allowed to be close to the obstacles, avoid and adjust
        if (not ignore_path_refs_idx_to_avoid) and (new_ref_idx in self.path_refs_idx_to_avoid):
            print(f"avoiding {new_ref_idx}")
            # copy the new reference into lower and upper references
            lower_ref_idx = new_ref_idx
            upper_ref_idx = new_ref_idx
            # keep adjusting the lower reference towards the vehicle 
            # till it is away from the obstacle or reached the track starting point or current vehicle position
            while True:
                lower_ref_idx -= 1
                if lower_ref_idx not in self.path_refs_idx_to_avoid or lower_ref_idx == 0 or lower_ref_idx == current_pos_idx:
                    break
            # keep adjusting the upper reference away from the vehicle
            # till it is away from the obstacle or reached the track end point or current vehicle position
            while True:
                upper_ref_idx += 1
                if upper_ref_idx not in self.path_refs_idx_to_avoid or upper_ref_idx == self.num_ref_track_samples or upper_ref_idx == current_pos_idx:
                    break
            
            # to avoid generating references that are too far ahead or very close to the vehicle
            # if difference between upper reference and computed new reference is less than half the prediction horizon, 
            # use upper reference as new reference
            if upper_ref_idx - new_ref_idx < self.N/2:
                if return_obstacle_in_range_alert:
                    return upper_ref_idx, *self.is_obstacle_in_range(current_pos_idx=current_pos_idx)
                return upper_ref_idx
            # else if difference between computed new reference and lower reference is less than half the prediction horizon, 
            # use lower reference as new reference
            elif new_ref_idx - lower_ref_idx < self.N/2:
                if return_obstacle_in_range_alert:
                    return lower_ref_idx, *self.is_obstacle_in_range(current_pos_idx=current_pos_idx)
                return lower_ref_idx
            # if all fails, default to using lower reference
            else:
                if lower_ref_idx == 0 or upper_ref_idx == self.num_ref_track_samples:
                    print(f"warning cannot avoid/ pass the obstacle at {new_ref_idx}\nnew reference index is {lower_ref_idx}")
                if return_obstacle_in_range_alert:
                    return lower_ref_idx, *self.is_obstacle_in_range(current_pos_idx=current_pos_idx)
                return lower_ref_idx
        if return_obstacle_in_range_alert:
            return new_ref_idx, *self.is_obstacle_in_range(current_pos_idx=current_pos_idx)
        return new_ref_idx

    def get_new_reference_point(self, current_state_flatened, enable_refer_fixed_num_points_ahead=False, 
                                motion_direction="ccw", ignore_path_refs_idx_to_avoid=False, 
                                return_obstacle_in_range_alert=False):
        """Compute and return a new reference point for the NMPC.
        
        :param current_state_flatened: current state of the vehicle dynamics.
        :param enable_refer_fixed_num_points_ahead: Boolean to indicate whether to compute a reference 
                            that is fixed number of points ahead of the vehicle's current position. Default False.
        :param motion_direction: direction of circular motion of the vehicle. Default "ccw".
        :param ignore_path_refs_idx_to_avoid: Boolean, whether to ignore generate references close to obstacles.
                                                True means references are allowed to be close to obstacles. Default False.
        :param return_obstacle_in_range_alert: Boolean, whether to return an alert if an obstacle is detected. Default False.
        :return tuple(self.reference_track_positions_nby2[new_ref_idx], is_obstacle_in_range, obstacles_in_range_ids) 
                                                                    or self.reference_track_positions_nby2[ref_idx]: Self-explanatory.
        """
        if return_obstacle_in_range_alert:
            ref_idx, is_obstacle_in_range, obstacles_in_range_ids = self.get_new_reference_point_idx(current_state_flatened, enable_refer_fixed_num_points_ahead, motion_direction, ignore_path_refs_idx_to_avoid, return_obstacle_in_range_alert)
            return self.reference_track_positions_nby2[ref_idx], is_obstacle_in_range, obstacles_in_range_ids
        ref_idx = self.get_new_reference_point_idx(current_state_flatened, enable_refer_fixed_num_points_ahead, motion_direction, ignore_path_refs_idx_to_avoid)
        return self.reference_track_positions_nby2[ref_idx]
            

            



            
            






