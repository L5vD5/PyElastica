""" Rendering Script using POVray

This script reads simulated data file to render POVray animation movie.
The data file should contain dictionary of positions vectors and times.

The script supports multiple camera position where a video is generated
for each camera view.

Notes
-----
    The module requires POVray installed.
"""

import sys

sys.path.append("/home/yoonbyeong/Dev/PyElastica/")

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm
import json

from examples.Visualization._povmacros import Stages, pyelastica_rod, render

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt

# Setup (USER DEFINE)
DATA_PATH = "rod_finger_4.dat"  # Path to the simulation data
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = "rod_finger_1"
OUTPUT_IMAGES_DIR = "frames_diag"
FPS = 20.0
WIDTH = 1920  # 400
HEIGHT = 1080  # 250
DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']

# Camera/Light Configuration (USER DEFINE)
stages = Stages()
stages.add_camera(
    # Add diagonal viewpoint
    location=[15.0, 10.5, 15.0],
    angle=30,
    look_at=[4.0, 2.7, 2.0],
    name="diag",
)
# stages.add_camera(
#     # Add top viewpoint
#     location=[0, 15, 3],
#     angle=30,
#     look_at=[0.0, 0, 3],
#     sky=[-1, 0, 0],
#     name="top",
# )
stages.add_light(
    # Sun light
    position=[1500, 2500, -1000],
    color="White",
    camera_id=-1,
)
stages.add_light(
    # Flash light for camera 0
    position=[15.0, 10.5, -15.0],
    color=[0.09, 0.09, 0.1],
    camera_id=0,
)
stages.add_light(
    # Flash light for camera 1
    position=[0.0, 8.0, 5.0],
    color=[0.09, 0.09, 0.1],
    camera_id=1,
)
stage_scripts = stages.generate_scripts()

# Externally Including Files (USER DEFINE)
# If user wants to include other POVray objects such as grid or coordinate axes,
# objects can be defined externally and included separately.
included = ["../default.inc"]

# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
NUM_AGENT = multiprocessing.cpu_count()  # number of parallel rendering.

# Execute
if __name__ == "__main__":
    # Load Data
    assert os.path.exists(DATA_PATH), "File does not exists"
    try:
        if SAVE_PICKLE:
            import pickle as pk

            with open(DATA_PATH, "rb") as fptr:
                data = pk.load(fptr)
        else:
            # (TODO) add importing npz file format
            raise NotImplementedError("Only pickled data is supported")
    except OSError as err:
        print("Cannot open the datafile {}".format(DATA_PATH))
        print(str(err))
        raise

    # Convert data to numpy array
    print(np.array(data["position"]).shape)
    times = np.array(data["time"])  # shape: (timelength)
    xs = np.array(data["position"])  # shape: (timelength, 3, num_element)

    # Interpolate Data
    # Interpolation step serves two purposes. If simulated frame rate is lower than
    # the video frame rate, the intermediate frames are linearly interpolated to
    # produce smooth video. Otherwise if simulated frame rate is higher than
    # the video frame rate, interpolation reduces the number of frame to reduce
    # the rendering time.
    runtime = times.max()  # Physical run time
    total_frame = int(runtime * FPS)  # Number of frames for the video
    recorded_frame = times.shape[0]  # Number of simulated frames
    times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale
    
    # print(xs.shape, times.shape)
    xs = interpolate.interp1d(times, xs, axis=0)(times_true)
    times = interpolate.interp1d(times, times, axis=0)(times_true)
    base_radius = np.ones_like(xs[:, 0, :]) * 0.050  # (TODO) radius could change
    
    # elastica data
    pos_elastica = xs[-1].swapaxes(0,1)
    print(pos_elastica)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.1,0.1)
    ax.set_zlim(-0.1,0.1)

    scatter = ax.scatter(pos_elastica[:,0]/10, pos_elastica[:,2]/10, pos_elastica[:,1]/10,  # 3D scatter plot
            s=10, alpha=0.5, c=range(len(pos_elastica[:,0])))
    for xi, yi, zi, pidi in zip(pos_elastica[:,0]/10,pos_elastica[:,2]/10,pos_elastica[:,1]/10, range(len(pos_elastica[:,0]))):
        # label = pidi
        ax.text(xi, yi, zi, None)

    # plt.savefig('plot.png')

    from matplotlib.animation import FuncAnimation

    def update(frame, pos_elastica, scatter):
        # print(pos_elastica.shape)
        scatter._offsets3d = pos_elastica[frame,0,:]/10, pos_elastica[frame,2,:]/10, pos_elastica[frame,1,:]/10
        print(pos_elastica[frame,:,:])
        # scatter.set_3d_properties(pos_elastica[:,1]/10)
        # line2.set_data(data2[frame, :, 0], data2[frame, :, 1])
        # line2.set_3d_properties(data2[frame, :, 2])
        return scatter

    print('xs shape : ', xs.shape)
    # ani = FuncAnimation(fig, update, fargs=[xs, scatter], frames=range(xs.shape[0]), interval=10)
    # ani.save('./animation.gif', writer='imagemagick', fps=60)
    plt.show()