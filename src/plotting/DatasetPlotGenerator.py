import os

import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
from src.enum.DatasetEnum import DatasetEnum

mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class DatasetPlotGenerator:
    def __init__(self, dataset_type, batch_size):
        self.batch_size = batch_size
        self.type = dataset_type

    def GeneratePlot(self, model, path_points, save_folder, step, future_steps):
        if self.type != DatasetEnum.Simulated:
            raise ValueError("Unknown dataset")

        step_save_path = save_folder + 'step{}/'.format(step)
        self.create_dir(step_save_path)

        grid_00, grid_01 = model.domain_descriptor.grid_domain[0]
        grid_10, grid_11 = model.domain_descriptor.grid_domain[1]

        # grid_10, grid_11 =0.85, 1.7
        # grid_00, grid_01 = 0.85, 1.7
        grid_10, grid_11 =0.5, 1.5
        grid_00, grid_01 = 0.5, 1.5

        XGrid = np.arange(grid_00, grid_01 - 1e-10, model.domain_descriptor.grid_gap)
        YGrid = np.arange(grid_10, grid_11 - 1e-10, model.domain_descriptor.grid_gap)
        grid_extent = [grid_00, grid_01, grid_11, grid_10 ]
        XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        # for max and min
        ground_truth_function = np.vectorize(lambda x, y: model([x, y]))
        ground_truth = ground_truth_function(XGrid, YGrid)
        vmax = np.amax(ground_truth)
        vmin = np.amin(ground_truth)


        is_eps = False

        self._generate_plot(model=model,
                            path_points=path_points,
                            step_save_path =step_save_path,
                            step=step,
                            future_steps=future_steps,
                            XGrid=XGrid, YGrid=YGrid, grid_extent=grid_extent,
                            is_eps=is_eps,
                            vmax=vmax,
                            vmin=vmin)

        for i in range(len(future_steps)):
            self._generate_posterior_mean(model=model,
                                          path_points=path_points,
                                          step_save_path=step_save_path,
                                          future_steps=future_steps,
                                          future_steps_it=i,
                                          XGrid=XGrid, YGrid=YGrid, grid_extent=grid_extent,
                                          is_eps=is_eps,
                                          vmax=vmax,
                                          vmin=vmin
                                          )

    @staticmethod
    def generate_posterior_history(model, path_points, future_steps, future_step_iteration):

        base_history = np.vstack(path_points)
        # print base_history
        base_measurements = np.apply_along_axis(lambda x: model(x), 1, base_history)
        # print base_measurements.shape
        for i in range(future_step_iteration-1):

            # Mle measurements for this iteration
            iteration_measurements = np.apply_along_axis(
                lambda x: model.gp.GPMean_without_weights(locations=base_history,
                                                          measurements=base_measurements,
                                                          current_location=x),
                1, future_steps[i]).flatten()

            base_history = np.append(base_history, future_steps[i], axis=0)
            base_measurements = np.append(base_measurements, iteration_measurements)
        # print base_history.shape, base_measurements.shape
        return base_history, base_measurements

    def _generate_posterior_mean(self, model, path_points, step_save_path,
                                 future_steps, future_steps_it,
                                 XGrid, YGrid, grid_extent,
                                 is_eps,
                                 vmax, vmin):

        base_history, base_measurements = self.generate_posterior_history(model=model,
                                                                          path_points=path_points,
                                                                          future_steps=future_steps,
                                                                          future_step_iteration=future_steps_it)

        base_history = base_history[: -self.batch_size, :]
        base_measurements = base_measurements[:-self.batch_size]

        ground_truth_function = np.vectorize(lambda x, y:
                                             model.gp.GPMean_without_weights(locations=base_history,
                                                                             measurements=base_measurements,
                                                                             current_location=[x, y]))
        ground_truth = ground_truth_function(XGrid, YGrid)

        gr = ground_truth.reshape(-1, 1)
        a = np.append(XGrid.reshape(-1, 1), YGrid.reshape(-1, 1), axis=1)
        total = np.append(a, gr, axis=1)


        axes = plt.axes()

        # current action
        prev = path_points[-2]
        current = path_points[-1]

        arrows = [tuple(map(tuple, prev))]

        self.draw_arrow(prev=prev, current=current, axes=axes)

        for i in range(future_steps_it):
            prev = future_steps[i]
            current = future_steps[i + 1]
            self.draw_arrow(prev=prev, current=current, axes=axes, lw=0.8, edgecolor='red')

        axes.imshow(ground_truth,
                    interpolation='bilinear',
                    aspect='auto',
                    # cmap='Greys',
                    cmap=cm.jet,
                    extent=grid_extent,
                    vmin=vmin, vmax=vmax)

        np.savetxt(fname=step_save_path + "step{}_mean_dataset.txt".format(future_steps_it), X=total,
                   fmt='%10.8f')

        output_file_name = step_save_path + "step{}_mean".format(future_steps_it)

        if is_eps:
            plt.savefig(output_file_name + ".eps", format='eps', dpi=1000, bbox_inches='tight')
        else:
            plt.savefig(output_file_name + ".png")
        plt.clf()
        plt.close()

    def _generate_plot(self, model, path_points,
                       step_save_path, step,
                       future_steps,
                       grid_extent, XGrid, YGrid,
                       is_eps,
                       vmax, vmin):
        # step_save_path = save_path + 'step{}/'.format(step)
        # self.create_dir(step_save_path)

        # grid_00, grid_01 = model.domain_descriptor.grid_domain[0]
        # grid_10, grid_11 = model.domain_descriptor.grid_domain[1]
        #
        # XGrid = np.arange(grid_00, grid_01 - 1e-10, model.domain_descriptor.grid_gap)
        # YGrid = np.arange(grid_10, grid_11 - 1e-10, model.domain_descriptor.grid_gap)
        #
        ground_truth_function = np.vectorize(lambda x, y: model([x, y]))

        # XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        # grid_extent = [grid_00, grid_01, grid_10, grid_11]

        ground_truth = ground_truth_function(XGrid, YGrid)

        axes = plt.axes()

        number_of_points = len(path_points)
        prev = path_points[-2]
        current = path_points[-1]

        self.draw_arrow(prev=prev, current=current, axes=axes)

        if step == 9:
            color = 'green'
        else:
            color = 'blue'

        # initial agent location
        first = path_points[0][-1]
        axes.plot(first[0], first[1], 'o', ms=5.0, color='green')

        for i in range(number_of_points - 2):
            prev = path_points[i]
            current = path_points[i + 1]
            self.draw_arrow(prev=prev, current=current, axes=axes, lw=1.0, edgecolor=color)

        future_steps_len = len(future_steps)
        for i in range(future_steps_len - 1):
            prev = future_steps[i]
            current = future_steps[i + 1]
            self.draw_arrow(prev=prev, current=current, axes=axes, lw=1.0, edgecolor='black')

        axes.imshow(ground_truth,
                    interpolation='bilinear',
                    aspect='auto',
                    # cmap='Greys',
                    cmap= cm.jet,
                    extent=grid_extent,
                    vmin=vmin, vmax=vmax)

        output_file_name = step_save_path + "step{}_result".format(step)
        if is_eps:
            plt.savefig(output_file_name + ".eps", format='eps', dpi=1000, bbox_inches='tight')
        else:
            plt.savefig(output_file_name + ".png")
        plt.clf()
        plt.close()

    # draw arrows from prev to current
    @staticmethod
    def draw_arrow(prev, current, axes, lw=2.0, edgecolor='green'):
        prev_end = prev[-1, :]
        # current_start = current[0, :]
        current_end = current[-1, :]
        axes.arrow(prev_end[0], prev_end[1],
                   current_end[0] - prev_end[0], current_end[1] - prev_end[1],
                   edgecolor=edgecolor, facecolor=edgecolor, lw=lw,
                   head_length=0.015, head_width=0.025, length_includes_head=True)

        # plt.annotate(s='', xy=(prev_end[0], prev_end[1]),
        #              xytext=(current_end[0] - prev_end[0], current_end[1] - prev_end[1]),
        #              arrowprops=dict(arrow style='->', edgecolor=edgecolor, facecolor=edgecolor, lw=lw))

        # here we need to draw k - 1 arrows
        # coz in total there will be k and the first on is already drawn

        # k should always be equal to batch_size though
        k = current.shape[0]

        # for j in xrange(0, k - 1):
        #     # both a locations [x,y]
        #     current_point = current[j, :]
        #     next_point = current[j + 1, :]
        #     axes.arrow(current_point[0], current_point[1],
        #                next_point[0] - current_point[0],
        #                next_point[1] - current_point[1], edgecolor=edgecolor, facecolor=edgecolor, lw=lw)


        for j in xrange(0, k):
            # both a locations [x,y]
            current_point = current[j, :]
            axes.plot(current_point[0], current_point[1], 'o',  ms=5.0, color=edgecolor)

    @staticmethod
    def create_dir(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            if not os.path.isdir(dir_name):
                raise
