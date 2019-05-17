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

    def GeneratePlot(self, model, path_points, save_path):
        if self.type == DatasetEnum.Robot:
            self.__generate_scatter_plot(model, path_points, save_path)
        elif self.type == DatasetEnum.Road:
            aspect = 2
            self.__generate_plot(model=model, path_points=path_points, save_path=save_path, aspect=aspect)
        elif self.type == DatasetEnum.Simulated:
            self.__generate_plot(model, path_points, save_path)

        else:
            raise ValueError("Unknown dataset")

    @staticmethod
    def __generate_scatter_plot(model, path_points, save_path):
        locations = model.locations
        values = model.values
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
    def __generate_plot(model, path_points, save_path, aspect=1):

        grid_00, grid_01 = model.domain_descriptor.grid_domain[0]
        grid_10, grid_11 = model.domain_descriptor.grid_domain[1]

        XGrid = np.arange(grid_00, grid_01 - 1e-10, model.domain_descriptor.grid_gap)
        YGrid = np.arange(grid_10, grid_11 - 1e-10, model.domain_descriptor.grid_gap)
        ground_truth_function = np.vectorize(lambda x, y: model([x, y]))

        XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        grid_extent = [grid_00, grid_01, grid_10, grid_11]

        ground_truth = ground_truth_function(XGrid, YGrid)

        axes = plt.axes()

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

        axes.imshow(ground_truth,
                    interpolation='nearest',
                    aspect='auto',
                    cmap='Greys',
                    extent=grid_extent)

        plt.savefig(save_path + ".png")
        plt.clf()
        plt.close()

