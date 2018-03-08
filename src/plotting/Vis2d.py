import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import cm


class Vis2d:
    def __init__(self):
        pass

    def MapPlot(self, locations, values, ground_truth=None, path_points=None, display=True,
                save_path=None):

        """
        grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                        grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left
        """
        X = locations[:, 0]
        Y = locations[:, 1]

        colors = ( values - np.mean(values) ) / np.std(values) * 100
        mmax = np.amax(np.amax(values))
        mmin = np.amin(np.amin(values))
        axes = plt.axes()

        axes.scatter(X, Y, s=30, c=values,  vmin=mmin, vmax=mmax, cmap = cm.jet)
        """
        im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto',
                             cmap='Greys', vmin=mmin, vmax=mmax)
        """
        if not path_points == None and path_points:
                # batch size
                # path points is a list
                number_of_points = len(path_points)

                for i in xrange(1, number_of_points):
                    # both are batches of points
                    prev = path_points[i-1]
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

                    for j in xrange(0, k-1):
                        # both a locations [x,y]
                        current_point = current[j, :]
                        next_point = current[j+1, :]
                        axes.arrow(current_point[0], current_point[1],
                                   next_point[0] - current_point[0],
                                   next_point[1] - current_point[1], edgecolor='red')


        if not save_path == None:
            plt.savefig(save_path + ".png")
        if display: plt.show()
        plt.clf()
        plt.close()

