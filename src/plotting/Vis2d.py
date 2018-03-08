import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Vis2d:
    def __init__(self):
        pass

    @staticmethod
    def MapPlot(locations, values, path_points, save_path):

        X = locations[:, 0]
        Y = locations[:, 1]

        mmax = np.amax(np.amax(values))
        mmin = np.amin(np.amin(values))
        axes = plt.axes()

        axes.scatter(X, Y, s=30, c=values, vmin=mmin, vmax=mmax, cmap=cm.jet)

        # batch size
        # path points is a list
        number_of_points = len(path_points)

        for i in xrange(1, number_of_points):
            # both are batches of points
            prev = path_points[i - 1]
            current = path_points[i]

            prev_end = prev[-1, :]
            current_start = current[0, :]
            axes.arrow(prev_end[0], prev_end[1],
                       current_start[0] - prev_end[0],
                       current_start[1] - prev_end[1], edgecolor='green')

            # here we need to draw k - 1 arrows
            # coz in total there will be k and the first on is already drawn

            # k should always be equal to batch_size though
            k = current.shape[0]

            for j in xrange(0, k - 1):
                # both a locations [x,y]
                current_point = current[j, :]
                next_point = current[j + 1, :]
                axes.arrow(current_point[0], current_point[1],
                           next_point[0] - current_point[0],
                           next_point[1] - current_point[1], edgecolor='red')

        plt.savefig(save_path + ".png")
        plt.clf()
        plt.close()
