import json
import numpy as np
import matplotlib.pyplot as plt
import os
# Opening JSON file
a = open(os.path.join(os.path.dirname(__file__), 'FINGER.json'))

dataset = json.load(a)
actuation = np.array(dataset['train']['motor_control'])[0]
position = np.array(dataset['train']['position']).squeeze(1)[0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-0.1,0.1)
ax.set_ylim(-0.1,0.1)
ax.set_zlim(-0.1,0.1)

scatter = ax.scatter(np.linspace(0,position[0],10), np.linspace(0,position[1],10), np.linspace(0,position[2],10), # 3D scatter plot
        s=10, alpha=0.5, c=range(10))
plt.title("actuation: [{}, {}]".format(str(actuation[0]), str(actuation[1])))

plt.savefig("plot_json.png")
