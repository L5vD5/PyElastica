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
import time
import pickle
import ray
from tqdm import tqdm
import random
import json
sys.path.append("../../..")  # isort:skip
sys.path.append("/home/yoonbyeong/Dev/PyElastica")
# from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from collections import defaultdict
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces, MuscleTorques, EndpointForces, UniformForces, UniformTorques
# from elastica.external_forces import InternalForces
from elastica.boundary_conditions import OneEndFixedRod
from elastica.interaction import AnisotropicFrictionalPlane
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate


class StretchingBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass



# Options
PLOT_FIGURE = False
SAVE_FIGURE = False
SAVE_RESULTS = True

snake_sim = StretchingBeamSimulator()

# setting up test params
n_elem = 9
start = np.array((0.0, 0.0, 0.0))
direction = np.array([0.0, 1.0, 0.0])
normal = np.array([1.0, 0.0, 0.0])
base_length = 1.2
base_radius = 0.1039
base_area = np.pi * base_radius ** 2
density = 1000
nu = 50.0 # 5.0
E = 3e4 #8753
poisson_ratio = 0.5

@ray.remote
def simulate(torques):
    snake_sim = StretchingBeamSimulator()

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

    final_time = 11.0

    # torques = [14.20097808, 5.76058777, 2.91217993, 7.76707398]
    # torques = [0.0, 0.0, 0.0, -10.0]

    # X axis
    snake_sim.add_forcing_to(shearable_rod).using(
        UniformTorques, torque = -torques[0], direction=np.array([-0.075, 0, 0.0375])
    )
    snake_sim.add_forcing_to(shearable_rod).using(
        UniformTorques, torque = +torques[0], direction=np.array([-0.075, 0, -0.0375])
    )

    # Z axis
    snake_sim.add_forcing_to(shearable_rod).using(
        UniformTorques, torque = +torques[1], direction=np.array([0.0375, 0, -0.075])
    )
    snake_sim.add_forcing_to(shearable_rod).using(
        UniformTorques, torque = -torques[1], direction=np.array([-0.0375, 0, -0.075])
    )

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
                # print(system.position_collection.copy())
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["directors"].append(
                    system.director_collection.copy()
                )
                # print(self.callback_params["directors"][-1][:,:,2])
                return

    period = 1.0

    pp_list = defaultdict(list)
    snake_sim.collect_diagnostics(shearable_rod).using(
        ContinuumSnakeCallBack, step_skip=200, callback_params=pp_list
    )

    snake_sim.finalize()
    timestepper = PositionVerlet()
    # timestepper = PEFRL()

    final_time = (final_time + 0.01) * period
    dt = 4.0e-5 * period
    total_steps = int(final_time / dt)
    # print("Total steps", total_steps)
    integrate(timestepper, snake_sim, final_time, total_steps, progress_bar=False)

    # filename = "rod_finger_{}.dat".format(4)
    # file = open(filename, "wb")
    # pickle.dump(pp_list, file)

    # file.close()
    
    return pp_list['position'][-1].swapaxes(0,1)

if __name__ == "__main__":
    ray.init()
    n_workers = 40
    torques = (np.random.rand(n_workers, 2)-1/2)*80
    num_dataset = 2000
    num_iter = num_dataset / n_workers
    motor_control_list = []
    position_list = []
    file_path = "./Elastica_Dataset.json"
    data = {}
    data['position'] = []
    data['motor_control'] = []


    for i in tqdm(range(int(num_iter))):
        futures = [simulate.remote(torques[i]) for i in range(n_workers)]
        results = ray.get(futures)
        [motor_control_list.append(torque.tolist()) for torque in torques]
        [position_list.append(result[:, [0, 2, 1]].tolist()[-1]) for result in results]
    
    data['position'] = position_list
    data['motor_control'] = motor_control_list
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

    
