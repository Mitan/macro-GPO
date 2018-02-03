import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import cm


def drawPlot(all_locations, values, path_points, save_path, current_step, batch_size):

    # coordinates
    X = all_locations[:, 0]
    Y = all_locations[:, 1]

    mmax = np.amax(np.amax(values))
    mmin = np.amin(np.amin(values))
    axes = plt.axes()

    axes.scatter(X, Y, s=30, c=values, vmin=mmin, vmax=mmax, cmap=cm.jet)

    # assert len(path_points) == 22
    start_point = path_points[0]
    fake_point = path_points[1]
    circle = plt.Circle(start_point, 0.04, lw=4.0, color='black', fill=False)
    axes.add_artist(circle)
    circle = plt.Circle(fake_point, 0.04, lw=2.0, color='black', fill=False)
    axes.add_artist(circle)
    colors = ['red', 'blue', 'green', 'black', 'brown']
    for i in range(current_step+1):
        # color the batch
        for j in range(batch_size):
            circle = plt.Circle(path_points[2 + i * batch_size + j], 0.04, lw=1.0, color=colors[i], fill=False)
            axes.add_artist(circle)

    plt.savefig(save_path + "step%d.png" % current_step)
    plt.clf()
    plt.close()
