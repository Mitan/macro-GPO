import matplotlib.pyplot as plt
from matplotlib  import cm
import numpy as np
"""
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("X vs Y AVG",fontsize=14)
ax.set_xlabel("XAVG",fontsize=12)
ax.set_ylabel("YAVG",fontsize=12)
ax.grid(True,linestyle='-',color='0.75')
x = np.random.random(30)
y = np.random.random(30)
z = np.random.random(30)

# scatter with colormap mapping to z value
ax.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.jet );

plt.show()

"""

"""
=================
An animated image
=================

This example demonstrates how to animate an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

ax = fig.gca()

# plt.scatter(20, 70,  color='black', marker='+', s=100)
# plt.plot(20, 70,  color='black', marker='+', markersize=12)
im = plt.imshow(f(x, y), animated=True)

x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
y = [50, 10, 40, 50, 70, 20, 15, 60, 20]
batch_size = 2
time_steps = 8
i = 0

# initila point + time_steps points + one for clearing
animation_stages = 1 + time_steps  + 1

def updatefig(*args):
    global i

    mod_i = i % (animation_stages)
    # for initial point and ends of macro-actions (visited locations)
    if mod_i % batch_size == 0:
        circle_patch = plt.Circle((x[mod_i], y[mod_i]), 2, color='black')
        ax.add_patch(circle_patch)
        plt.pause(1.5)

    # last stage for clearing
    if mod_i == animation_stages - 1:
        plt.pause(2.0)
        # arrows + circles at visited locations + initial point
        num_patches = time_steps + time_steps/batch_size + 1
        ax.patches = ax.patches[:-num_patches]

    # las location doesn't have an arrow
    elif mod_i == animation_stages - 2:
        pass

    else:
        patch = plt.Arrow(x[mod_i] , y[mod_i] , x[mod_i+1] - x[mod_i], y[mod_i+1] - y[mod_i], color='black')
        ax.add_patch(patch)
        plt.pause(1.0)

    i+= 1

    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
plt.show()
ani.save('line.gif', dpi=80, writer='imagemagick')