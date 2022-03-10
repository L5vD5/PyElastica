""" Axial stretching test-case

    Assume we have a rod lying aligned in the x-direction, with high internal
    damping.

    We fix one end (say, the left end) of the rod to a wall. On the right
    end we apply a force directed axially pulling the rods tip. Linear
    theory (assuming small displacements) predict that the net displacement
    experienced by the rod tip is Δx = FL/AE where the symbols carry their
    usual meaning (the rod is just a linear spring). We compare our results
    with the above result.

    We can "improve" the theory by having a better estimate for the rod's
    spring constant by assuming that it equilibriates under the new position,
    with
    Δx = F * (L + Δx)/ (A * E)
    which results in Δx = (F*l)/(A*E - F). Our rod reaches equilibrium wrt to
    this position.

    Note that if the damping is not high, the rod oscillates about the eventual
    resting position (and this agrees with the theoretical predictions without
    any damping : we should see the rod oscillating simple-harmonically in time).

    isort:skip_file
"""
# FIXME without appending sys.path make it more generic
import sys

sys.path.append("../../../")  # isort:skip
# from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from collections import defaultdict
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces, MuscleTorques, EndpointForces
from elastica.boundary_conditions import OneEndFixedRod
from elastica.interaction import AnisotropicFrictionalPlane
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate


class StretchingBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


final_time = 20.0

# Options
PLOT_FIGURE = False
SAVE_FIGURE = False
SAVE_RESULTS = True

snake_sim = StretchingBeamSimulator()

# setting up test params
n_elem = 20
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.0
base_radius = 0.025
base_area = np.pi * base_radius ** 2
density = 1000
nu = 5.0
E = 1e7
poisson_ratio = 0.5

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    poisson_ratio,
)

snake_sim.append(shearable_rod)

snake_sim.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

end_force_x = 10.0

end_force = np.array([end_force_x, end_force_x, -end_force_x])
snake_sim.add_forcing_to(shearable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=1e-2
)

# Add gravitational forces
# gravitational_acc = -9.80665
# snake_sim.add_forcing_to(shearable_rod).using(
#     GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
# )

period = 1.0
# b_coeff = np.array([17.4, 48.5, 5.4, 14.7, 0.97])
# wave_length = b_coeff[-1]
# snake_sim.add_forcing_to(shearable_rod).using(
#     MuscleTorques,
#     base_length=base_length,
#     b_coeff=b_coeff[:-1],
#     period=period,
#     wave_number=2.0 * np.pi / (wave_length),
#     phase_shift=0.0,
#     direction=normal,
#     rest_lengths=shearable_rod.rest_lengths,
#     ramp_up_time=period,
#     with_spline=True,
# )

# # Add friction forces
# origin_plane = np.array([0.0, -base_radius, 0.0])
# normal_plane = normal
# slip_velocity_tol = 1e-8
# froude = 0.1
# mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
# kinetic_mu_array = np.array(
#     [mu, 1.5 * mu, 2.0 * mu]
# )  # [forward, backward, sideways]
# static_mu_array = 2 * kinetic_mu_array
# snake_sim.add_forcing_to(shearable_rod).using(
#     AnisotropicFrictionalPlane,
#     k=1.0,
#     nu=1e-6,
#     plane_origin=origin_plane,
#     plane_normal=normal_plane,
#     slip_velocity_tol=slip_velocity_tol,
#     static_mu_array=static_mu_array,
#     kinetic_mu_array=kinetic_mu_array,
# )

# Add call backs
class ContinuumSnakeCallBack(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            print(system.position_collection.copy())
            self.callback_params["position"].append(
                system.position_collection.copy()
            )

            return

pp_list = defaultdict(list)
snake_sim.collect_diagnostics(shearable_rod).using(
    ContinuumSnakeCallBack, step_skip=200, callback_params=pp_list
)

snake_sim.finalize()
timestepper = PositionVerlet()
# timestepper = PEFRL()

final_time = (11.0 + 0.01) * period
dt = 5.0e-5 * period
total_steps = int(final_time / dt)
print("Total steps", total_steps)
integrate(timestepper, snake_sim, final_time, total_steps)

import pickle

filename = "axial_stretching_diag.dat"
file = open(filename, "wb")
# pickle.dump(stretchable_rod, file)
pickle.dump(pp_list, file)

file.close()
