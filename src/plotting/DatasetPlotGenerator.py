import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
from src.enum.DatasetEnum import DatasetEnum

mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class DatasetPlotGenerator:
    def __init__(self, dataset_type):
        self.type = dataset_type

    def GeneratePlot(self, locations, values, path_points, save_path):
        if self.type == DatasetEnum.Robot:
            self.__generate_robot_plot(locations, values, path_points, save_path)
        elif self.type == DatasetEnum.Road:
            self.__generate_road_plot(locations, values, path_points, save_path)

        else:
            raise ValueError("Unknown dataset")

    @staticmethod
    def __generate_robot_plot(locations, values, path_points, save_path):

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

    @staticmethod
    # def __generate_road_plot(locations, values, path_points, save_path):
    def MapPlot(self, grid_extent, ground_truth=None, posterior_mean_before=None, posterior_mean_after=None,
                path_points=None, display=True,
                save_path=None):
        """
        Plots original field and path taken, as well as
        Saves data to file if required
        @param ground_truth, posterior mean, posterior variance - 2d-array of relvant data.
        - Each 2d array represents 1 field (eg. posterior, ground truth)
        - Note that these are all indexed in integers etc
        - We will need to scale and translate accordingly
        @param grid_extent - axis mesh points as a 4-tuple of numpy arrays comprising (x-min, xmax, ymin, ymax)
        @param path_points - path coordinates in "world" space (ie. actual coordinates, not just (5,3) etc ... )
        @param display - shows plot on screen, useful for debugging
        @param save_path
        """

        grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                        grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

        mmax = -10 ** 10
        mmin = 10 ** 10
        for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
            if q is not None:
                mmax = max(np.amax(np.amax(q)), mmax)
                mmin = min(np.amin(np.amin(q)), mmin)
        axes = plt.axes()
        # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        if ground_truth is not None:
            im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                             cmap='Greys', vmin=mmin, vmax=mmax)
            if not path_points == None and path_points:
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

        if not save_path == None:
            plt.savefig(save_path + ".png")
        if display: plt.show()
        plt.clf()
        plt.close()