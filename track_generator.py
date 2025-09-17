import numpy as np
from matplotlib.patches import Ellipse

# Only implementing eliptical tracks for now
class TrackGenerator:
    """Implements road track generation for simulation and visualisation purposes.
    
    For simulation, provides methods to generate elliptical tracks of required size 
        and samples the track boundaries (edges) and the centre of the track in to 
        desired number of samples.

    For track visualisation, provides methods to generate matplotlib elliptical patches.

    Also provides methods used for generating elliptical obstacle boundaries (safety circles).
    """
    def __init__(self, inner_track_radii, outer_track_radii, 
                 num_sample_positions_centre_track_per_meter=1, num_track_samples_per_meter=1):
        """Constructor.
        
        :param inner_track_radii: list of major and minor [a, b] radii of road track's inner boundary.
        :param outer_track_radii: list of major and minor [a, b] radii of road track's outer boundary.
        :param num_sample_positions_centre_track_per_meter: number of equidistant samples per meter of 
                                                                centre of the track to be generated. Default 1.
        :param num_track_samples_per_meter: number of equidistant samples per meter of track boundaries to be generated. Default 1.
        """
        self.num_track_samples_per_meter = num_track_samples_per_meter
        self.num_sample_positions_centre_track_per_meter = num_sample_positions_centre_track_per_meter
        
        self.global_track_centre = [0, 0]
        inner_track_data = self.generate_elliptical_boundaries(ellipse_radii_ab=inner_track_radii, 
                                                               ellipse_centre_pos_xy=self.global_track_centre,
                                                               num_boundary_samples_per_meter=self.num_track_samples_per_meter)
        self.inner_track_a = inner_track_data[0]
        self.inner_track_b = inner_track_data[1]
        self.inner_circumference_approx = inner_track_data[2]
        self.num_inner_track_points = inner_track_data[3]
        self.track_inner_edge_angles = inner_track_data[4]
        self.track_inner_edge_x = inner_track_data[5]
        self.track_inner_edge_y = inner_track_data[6]
        self.track_inner_edge = inner_track_data[7]
        self.track_inner_patch = Ellipse(self.global_track_centre, 2* self.inner_track_a, 2* self.inner_track_b,
                                         angle=0, fc="white")

        outer_track_data = self.generate_elliptical_boundaries(ellipse_radii_ab=outer_track_radii,
                                                               ellipse_centre_pos_xy=[0, 0],
                                                               num_boundary_samples_per_meter=self.num_track_samples_per_meter)
        self.outer_track_a = outer_track_data[0]
        self.outer_track_b = outer_track_data[1]
        self.outer_circumference_approx = outer_track_data[2]
        self.num_outer_track_points = outer_track_data[3]
        self.track_outer_edge_angles = outer_track_data[4]
        self.track_outer_edge_x = outer_track_data[5]
        self.track_outer_edge_y = outer_track_data[6]
        self.track_outer_edge = outer_track_data[7]
        self.track_outer_patch = Ellipse(self.global_track_centre, 2 * self.outer_track_a, 2 * self.outer_track_b,
                                         angle=0, fc="black")

        self.centre_track_a = (self.inner_track_a + self.outer_track_a)/2
        self.centre_track_b = (self.inner_track_b + self.outer_track_b)/2
        centre_track_data = self.generate_elliptical_boundaries(ellipse_radii_ab=[self.centre_track_a, self.centre_track_b],
                                                                ellipse_centre_pos_xy=[0, 0],
                                                                num_boundary_samples_per_meter=self.num_sample_positions_centre_track_per_meter)
        self.centre_circumference_approx = centre_track_data[2]
        self.num_centre_track_points = centre_track_data[3]
        self.centre_track_sample_angles = centre_track_data[4]
        self.track_centre_edge_x = centre_track_data[5]
        self.track_centre_edge_y = centre_track_data[6]
        self.track_centre_edge = centre_track_data[7]
        self.track_centre_patch = Ellipse(self.global_track_centre, 2 * self.centre_track_a, 2 * self.centre_track_b,
                                         angle=0, edgecolor="yellow", fc="black")
        
    def generate_elliptical_boundaries(self, ellipse_radii_ab: list, ellipse_centre_pos_xy: list, 
                                       num_boundary_samples_per_meter=10, rotate_major_axis_by_rad=0):
        """Generic method to generate and sample elliptical boundaries of desired dimensions.
        
        :param ellipse_radii_ab: list of major and minor radii [a, b] of the desired elliptical boundary.
        :param ellipse_centre_pos_xy: list of centre [cx, cy] of the desired elliptical boundary.
        :param num_boundary_samples_per_meter: number of samples to be taken of the ellipse per meter. Default 10 samples/m.
        :param rotate_major_axis_by_rad: slope of the ellipse's major axis in radians. Default 0 rad.
        :return: data list of the generated ellipse of form 
                    [
                        a, b, approx circumference, no. of samples, sampled angles in range [0, 2pi], 
                        boundaries' x coordinates, boundaries' y coordinates, 
                        combined list of x and y boundary coordinates
                    ]
        """
        ellipse_radius_a = np.max(ellipse_radii_ab)
        ellipse_radius_b = np.min(ellipse_radii_ab)
        ellipse_circumference_approx = np.pi*(ellipse_radius_a + ellipse_radius_b)
        num_sample_angles = int(ellipse_circumference_approx * num_boundary_samples_per_meter)
        sample_angles = np.linspace(0, 2*np.pi, num_sample_angles).reshape(-1, 1)
        x = ellipse_radius_a * np.cos(sample_angles) * np.cos(rotate_major_axis_by_rad) \
            - ellipse_radius_b * np.sin(sample_angles) * np.sin(rotate_major_axis_by_rad) \
                + ellipse_centre_pos_xy[0]
        y = ellipse_radius_b * np.sin(sample_angles) * np.cos(rotate_major_axis_by_rad) \
             + ellipse_radius_a * np.cos(sample_angles) * np.sin(rotate_major_axis_by_rad) \
                + ellipse_centre_pos_xy[1]
        ellipse_boundaries = np.concatenate([x, y], axis=1)
        return [ellipse_radius_a, ellipse_radius_b, ellipse_circumference_approx, 
                num_sample_angles, sample_angles, x, y, ellipse_boundaries]

    def generate_obstacle_ellipse(self, obstacle_radii_ab: list, obstacle_pos_xy: list, 
                                  num_obstacle_samples_per_meter=10, rotate_major_axis_by_rad=0):
        """Method to generate obstacle ellipse (safety circle).
        
        :param obstacle_radii_ab: list of major and minor radii [a, b] of the desired obstacle safety circle.
        :param obstacle_pos_xy: list of centre [cx, cy] of the desired obstacle safety circle.
        :param num_obstacle_samples_per_meter: number of samples to be taken of the safety circle per meter. Default 10 samples/m.
        :param rotate_major_axis_by_rad: slope of the safety circle's major axis in radians. Default 0 rad.
        :return: data tuple of the generated ellipse of form (combined list of x and y boundary coordinates, obstacle data).
                 obstacle data is a dict of form
                    {
                        "radii": [a, b], 
                        "centre": [cx, cy],
                        "num_samples": param num_obstacle_samples_per_meter,
                        "major_ax_rot_rad" = param rotate_major_axis_by_rad,
                        "circumference_approx" = approx circumference of the safety circle,
                        "num_sample_angles" = no. of samples,
                        "sample_angles" = sampled angles in range [0, 2pi],
                        "boundaries" = combined list of x and y boundary coordinates                        
                    } **make this a data class later.
        """
        obstacle_boundaries = self.generate_elliptical_boundaries(ellipse_radii_ab=obstacle_radii_ab, 
                                                                  ellipse_centre_pos_xy=obstacle_pos_xy,
                                                                  num_boundary_samples_per_meter=num_obstacle_samples_per_meter,
                                                                  rotate_major_axis_by_rad=rotate_major_axis_by_rad)
        obstacle_data = dict()
        obstacle_data["radii"] = obstacle_radii_ab
        obstacle_data["centre"] = obstacle_pos_xy
        obstacle_data["num_samples"] = num_obstacle_samples_per_meter
        obstacle_data["major_ax_rot_rad"] = rotate_major_axis_by_rad
        obstacle_data["circumference_approx"] = obstacle_boundaries[2]
        obstacle_data["num_sample_angles"] = obstacle_boundaries[3]
        obstacle_data["sample_angles"] = obstacle_boundaries[4]
        obstacle_data["boundaries"] = obstacle_boundaries[7]
        return obstacle_boundaries[7], obstacle_data
    
    def add_track_patch_to_plt_axs(self, ax):
        """Adds the generated road track patches to a given matplotlib axes object.
        
        :param ax: axes object of the matplotlib plot on which the road track needs to be visualised.
        """
        ax.add_patch(self.track_outer_patch)
        ax.add_patch(self.track_centre_patch)
        ax.add_patch(self.track_inner_patch)
    
    def slope_ellipse_rad(self, xy, ab):
        """Method to compute the slope of a tangent at a point on an ellipse.
        
        :param xy: list of point [x, y] at which the tangent is located on the ellipse.
        :param ab: list of major and minor [a, b] axis radii of the ellipse.
        :return: the slope of the tangent at [x, y] in radians."""
        return np.arctan2(- xy[0] * ab[1]**2, xy[1] * ab[0]**2)
