import numpy as np
import sys

sys.path.append("../../../")
from elastica import *
from post_processing import plot_velocity, plot_video_with_surface


def rod_cylinder_contact_friction_case(
    force_coefficient=0.1, normal_force_mag=10, POST_PROCESSING=False
):
    class RodCylinderParallelContact(
        BaseSystemCollection, Constraints, Connections, CallBacks, Forcing
    ):
        pass

    rod_cylinder_parallel_contact_simulator = RodCylinderParallelContact()

    # time step etc
    final_time = 20.0
    time_step = 1e-4
    total_steps = int(final_time / time_step) + 1
    rendering_fps = 30  # 20 * 1e1
    step_skip = int(1.0 / (rendering_fps * time_step))

    base_length = 0.5
    base_radius = 0.1
    density = 1750
    E = 3e5
    poisson_ratio = 0.5
    shear_modulus = E / (2 * (1 + poisson_ratio))
    n_elem = 50
    nu = 0.5
    start = np.zeros((3,))
    direction = np.array([0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])

    rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=shear_modulus,
    )

    rod_cylinder_parallel_contact_simulator.append(rod)

    # Push the rod towards the cylinder to make sure contact is there
    normal_force_direction = np.array([-1.0, 0.0, 0.0])
    rod_cylinder_parallel_contact_simulator.add_forcing_to(rod).using(
        UniformForces, force=normal_force_mag, direction=normal_force_direction
    )
    # Apply uniform forces on the rod
    rod_cylinder_parallel_contact_simulator.add_forcing_to(rod).using(
        UniformForces, force=normal_force_mag * force_coefficient, direction=direction
    )

    cylinder_height = 8 * base_length
    cylinder_radius = base_radius

    cylinder_start = start + np.array([-1.0, 0.0, 0.0]) * 2 * base_radius
    cylinder_direction = np.array([0.0, 0.0, 1.0])
    cylinder_normal = np.array([0.0, 1.0, 0.0])

    rigid_body = Cylinder(
        start=cylinder_start,
        direction=cylinder_direction,
        normal=cylinder_normal,
        base_length=cylinder_height,
        base_radius=cylinder_radius,
        density=density,
    )
    rod_cylinder_parallel_contact_simulator.append(rigid_body)

    # Constrain the rigid body position and directors
    rod_cylinder_parallel_contact_simulator.constrain(rigid_body).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    # Add contact between rigid body and rod
    rod_cylinder_parallel_contact_simulator.connect(rod, rigid_body).using(
        ExternalContact,
        k=1e5,
        nu=100,
        velocity_damping_coefficient=1e5,
        friction_coefficient=0.5,
    )

    # Add callbacks
    post_processing_dict_list = []
    # For rod
    class StraightRodCallBack(CallBackBaseClass):
        """
        Call back function for two arm octopus
        """

        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )
                if current_step == 0:
                    self.callback_params["lengths"].append(system.rest_lengths.copy())
                else:
                    self.callback_params["lengths"].append(system.lengths.copy())

                self.callback_params["com_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )

                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                    + system.compute_bending_energy()
                    + system.compute_shear_energy()
                )
                self.callback_params["total_energy"].append(total_energy)

                return

    class RigidCylinderCallBack(CallBackBaseClass):
        """
        Call back function for two arm octopus
        """

        def __init__(
            self, step_skip: int, callback_params: dict, resize_cylinder_elems: int
        ):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params
            self.n_elem_cylinder = resize_cylinder_elems
            self.n_node_cylinder = self.n_elem_cylinder + 1

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)

                cylinder_center_position = system.position_collection
                cylinder_length = system.length
                cylinder_direction = system.director_collection[2, :, :].reshape(3, 1)
                cylinder_radius = system.radius

                # Expand cylinder data. Create multiple points on cylinder later to use for rendering.

                start_position = (
                    cylinder_center_position - cylinder_length / 2 * cylinder_direction
                )

                cylinder_position_collection = (
                    start_position
                    + np.linspace(0, cylinder_length[0], self.n_node_cylinder)
                    * cylinder_direction
                )
                cylinder_radius_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_radius
                )
                cylinder_length_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_length
                )
                cylinder_velocity_collection = (
                    np.ones((self.n_node_cylinder)) * system.velocity_collection
                )

                self.callback_params["position"].append(
                    cylinder_position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    cylinder_velocity_collection.copy()
                )
                self.callback_params["radius"].append(cylinder_radius_collection.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )

                self.callback_params["lengths"].append(
                    cylinder_length_collection.copy()
                )
                self.callback_params["com_velocity"].append(
                    system.velocity_collection[..., 0].copy()
                )

                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                )
                self.callback_params["total_energy"].append(total_energy[..., 0].copy())

                return

    if POST_PROCESSING:
        post_processing_dict_list.append(defaultdict(list))
        rod_cylinder_parallel_contact_simulator.collect_diagnostics(rod).using(
            StraightRodCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_list[0],
        )
        # For rigid body
        post_processing_dict_list.append(defaultdict(list))
        rod_cylinder_parallel_contact_simulator.collect_diagnostics(rigid_body).using(
            RigidCylinderCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_list[1],
            resize_cylinder_elems=n_elem,
        )

    rod_cylinder_parallel_contact_simulator.finalize()
    timestepper = PositionVerlet()

    integrate(
        timestepper, rod_cylinder_parallel_contact_simulator, final_time, total_steps
    )

    if POST_PROCESSING:
        # Plot the rods
        plot_video_with_surface(
            post_processing_dict_list,
            video_name="rod_cylinder_contact.mp4",
            fps=rendering_fps,
            step=1,
            # The following parameters are optional
            x_limits=(-base_length * 5, base_length * 5),  # Set bounds on x-axis
            y_limits=(-base_length * 5, base_length * 5),  # Set bounds on y-axis
            z_limits=(-base_length * 5, base_length * 5),  # Set bounds on z-axis
            dpi=100,  # Set the quality of the image
            vis3D=True,  # Turn on 3D visualization
            vis2D=True,  # Turn on projected (2D) visualization
        )

        filaname = "rod_rigid_velocity.png"
        plot_velocity(
            post_processing_dict_list[0],
            post_processing_dict_list[1],
            filename=filaname,
            SAVE_FIGURE=True,
        )

    # Compute final total energy
    total_final_energy = (
        rod.compute_translational_energy()
        + rod.compute_rotational_energy()
        + rod.compute_bending_energy()
        + rod.compute_shear_energy()
    )

    return total_final_energy
