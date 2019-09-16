import os

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

    def GeneratePlot(self, model, path_points, save_folder, step, future_steps):
        if self.type == DatasetEnum.Simulated:
            self._generate_plot(model=model,
                                path_points=path_points,
                                save_path=save_folder,
                                step=step,
                                future_steps=future_steps)
        else:
            raise ValueError("Unknown dataset")

    def generate_posteriors(self, model, path_points, step_save_path, future_steps, future_step_iteration):
        base_history = np.vstack(path_points)
        # print base_history.shape
        base_measurements = np.apply_along_axis(lambda x: model(x), 1, base_history)
        # print base_measurements.shape
        for i in range(future_step_iteration):
            # Mle measurements for this iteration
            iteration_measurements = np.apply_along_axis(
                lambda x: model.gp.GPMean_without_weights(locations=base_history,
                                                          measurements=base_measurements,
                                                          current_location=x),
                1, future_steps[i]).flatten()

            base_history =  np.append(base_history, future_steps[i], axis=0)
            base_measurements = np.append(base_measurements, iteration_measurements)
            print base_measurements.shape, base_history.shape
        print a
        # posterior_mean_after = np.vectorize(
        #     lambda x, y: self.gp.GPMean(state_history[-1].history.locations, state_history[-1].history.measurements,
        #                                 [x, y]))

    def _generate_plot(self, model, path_points, save_path, step, future_steps, aspect=1):
        step_save_path = save_path + 'step{}/'.format(step)
        self.create_dir(step_save_path)

        future_steps_it = 0
        if step == 3:
            self.generate_posteriors(model=model,
                                     path_points=path_points,
                                     step_save_path=step_save_path,
                                     future_steps=future_steps,
                                     future_step_iteration=future_steps_it)

        grid_00, grid_01 = model.domain_descriptor.grid_domain[0]
        grid_10, grid_11 = model.domain_descriptor.grid_domain[1]

        XGrid = np.arange(grid_00, grid_01 - 1e-10, model.domain_descriptor.grid_gap)
        YGrid = np.arange(grid_10, grid_11 - 1e-10, model.domain_descriptor.grid_gap)
        ground_truth_function = np.vectorize(lambda x, y: model([x, y]))

        XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        grid_extent = [grid_00, grid_01, grid_10, grid_11]

        ground_truth = ground_truth_function(XGrid, YGrid)

        axes = plt.axes()

        number_of_points = len(path_points)
        prev = path_points[-2]
        current = path_points[-1]

        self.draw_arrow(prev=prev, current=current, axes=axes)

        for i in range(number_of_points - 2):
            prev = path_points[i]
            current = path_points[i + 1]
            self.draw_arrow(prev=prev, current=current, axes=axes, lw=0.8)

        future_steps_len = len(future_steps)
        for i in range(future_steps_len - 1):
            prev = future_steps[i]
            current = future_steps[i + 1]
            self.draw_arrow(prev=prev, current=current, axes=axes, lw=0.8, edgecolor='red')

        axes.imshow(ground_truth,
                    interpolation='nearest',
                    aspect='auto',
                    cmap='Greys',
                    extent=grid_extent)

        plt.savefig(step_save_path + "step{}_result.png".format(step))
        plt.clf()
        plt.close()

    # draw arrows from prev to current
    @staticmethod
    def draw_arrow(prev, current, axes, lw=2.0, edgecolor='green'):

        prev_end = prev[-1, :]
        current_start = current[0, :]
        axes.arrow(prev_end[0], prev_end[1],
                   current_start[0] - prev_end[0],
                   current_start[1] - prev_end[1], edgecolor=edgecolor, facecolor=edgecolor, lw=lw)

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
                       next_point[1] - current_point[1], edgecolor=edgecolor, facecolor=edgecolor, lw=lw)

    @staticmethod
    def create_dir(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            if not os.path.isdir(dir_name):
                raise
