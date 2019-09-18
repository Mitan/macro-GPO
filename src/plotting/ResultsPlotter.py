import matplotlib
import numpy as np

from src.enum.PlottingEnum import PlottingEnum

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
# from matplotlib import rc

# for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=True)


class ResultGraphPlotter:

    def __init__(self, dataset_type, batch_size, total_budget):
        self.total_budget = total_budget

        self.dataset_type = dataset_type

        # size of font at x and y label
        self.labels_font_size = 23

        # + 1 because of initial point
        plotting_num_steps = self.total_budget / batch_size + 1

        self.samples_collected = [i * batch_size for i in range(plotting_num_steps)]

        self.color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                               '#ff7f00', 'black', '#a65628', '#f781bf', 'blue']

        self.markers = ["o", "v", "^", "s", "*", "1", "2", "x", "|"]

    def plot_results(self, results, plot_bars, output_file_name, plotting_type):
        # param_storer = self._get_param_storer(plotting_type)

        if not results:
            return

        # 0 is width, 1 is height
        plt.rcParams["figure.figsize"] = [6, 9]

        # for legends
        handles = []

        for i, result in enumerate(results):
            handle = self.__plot_one_method(i, result, plot_bars)
            handles.append(handle)

        plt.legend(handles=handles, loc=0, prop={'size': 14})
        max_y_value = max([max(result[1]) for result in results])
        min_y_value = min([min(result[1]) for result in results])

        self.__ticks_and_margins(min_y_value=min_y_value,
                                 max_y_value=max_y_value,
                                 plotting_type=plotting_type)

        plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')
        plt.clf()
        plt.close()

    def __ticks_and_margins(self, min_y_value, max_y_value, plotting_type):
        plt.xticks(self.samples_collected)
        plt.xlabel("No. of observations", fontsize=self.labels_font_size)
        axes = plt.axes()
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # plt.ylabel(param_storer.y_label_caption, fontsize=self.labels_font_size)
        if plotting_type == PlottingEnum.AverageTotalReward:
            y_label_caption = "Average normalized output measurements"
        elif plotting_type == PlottingEnum.SimpleRegret:
            y_label_caption = "Simple Regret"
        else:
            raise ValueError("Unknown plotting type")

        plt.ylabel(y_label_caption, fontsize=self.labels_font_size)

        ticks_interval = 0.1

        y_ticks_min = round(min_y_value / ticks_interval)
        y_ticks_max = round(max_y_value / ticks_interval + 1)

        y_ticks = ticks_interval * np.arange(y_ticks_min, y_ticks_max)
        plt.yticks(y_ticks)

        tick_size = 15
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)

        # margins on x and y side
        axes.margins(x=0.035)
        axes.margins(y=0.035)

    # plot a method and return a legend handle
    def __plot_one_method(self, i, result, plot_bars):
        # result[0] is a MethodDescriptor
        name = result[0].latex_method_name

        rewards = result[1]
        error_bars = result[2]

        # hack for EI
        single_point_methods = len(rewards) == self.total_budget + 1
        adjusted_time_steps = range(self.total_budget + 1) if single_point_methods else self.samples_collected
        # marker_size = 10 if single_point_methods else 20
        marker_size = 20

        if plot_bars:
            marker_size = 5

        line_style = '-' if i < 8 else '--'
        marker_index = i

        # dirty hack to make it unfilled
        plt.plot(adjusted_time_steps, rewards, lw=1.0, linestyle=line_style, marker=self.markers[marker_index],
                 markersize=marker_size,
                 markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=self.color_sequence[i], color=self.color_sequence[i])

        if plot_bars:
            # if plot_bars and error_bars:

            plt.errorbar(adjusted_time_steps, rewards, yerr=error_bars, color=self.color_sequence[i], lw=0.1)

        patch = mlines.Line2D([], [], linestyle=line_style, color=self.color_sequence[i],
                              marker=self.markers[marker_index],
                              markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=self.color_sequence[i],
                              markersize=10, label=name)

        return patch
