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
i = 0
ax = fig.gca()
patch = plt.Arrow(0, 0, 90, 40)
ax.add_patch(patch)

im = plt.imshow(f(x, y), animated=True)


colors = ['green', 'red']
def updatefig(*args):
    global x, y,i
    x += np.pi / 15.
    y += np.pi / 20.
    i+=1
    # im.set_array(f(x, y))
    mod_i = i % 3
    if i % 3 == 2:
        ax.patches = ax.patches[:-2]
    else:
        patch = plt.Arrow(20 * mod_i , 20 * mod_i , 20  + 10, 20 * mod_i  + 10, color='black')
        ax.add_patch(patch)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=500, blit=True)
#plt.show()
ani.save('line.gif', dpi=80, writer='imagemagick')