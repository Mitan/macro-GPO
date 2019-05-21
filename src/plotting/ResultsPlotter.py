import matplotlib

from src.enum.DatasetEnum import DatasetEnum
from src.plotting.RoadDatasetPlotParamStorer import RoadDatasetPlotParamStorer
from src.plotting.RobotDatasetPlotParamStorer import RobotDatasetPlotParamStorer
from src.plotting.SimulatedDatasetPlotParamStorer import SimulatedDatasetPlotParamStorer

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc

# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


class ResultGraphPlotter:

    def __init__(self, dataset_type, plotting_type, batch_size, total_budget):
        self.total_budget = total_budget
        self.plotting_type = plotting_type
        self.dataset_type = dataset_type
        self.param_storer = self._get_param_storer()

        # + 1 because of initial point
        plotting_num_steps = self.total_budget / batch_size + 1

        self.samples_collected = [i * batch_size for i in range(plotting_num_steps)]

        self.color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                               '#ff7f00', 'black', '#a65628', '#f781bf', 'blue']

        self.markers = ["o", "v", "^", "s", "*", "1", "2", "x", "|"]

    def plot_results(self, results, plot_bars, output_file_name):
        if not results:
            return

        # 0 is width, 1 is height
        plt.rcParams["figure.figsize"] = [6, 9]

        # for legends
        handles = []

        for i, result in enumerate(results):
            handle = self.__plot_one_method(i, result, plot_bars)
            handles.append(handle)

        plt.legend(handles=handles, loc=self.param_storer.legend_loc, prop={'size': self.param_storer.legend_size})

        self.__ticks_and_margins()

        plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')
        plt.clf()
        plt.close()

    def __ticks_and_margins(self):
        plt.xticks(self.samples_collected)

        plt.xlabel("No. of observations", fontsize=30)
        axes = plt.axes()
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.ylabel(self.param_storer.y_label_caption, fontsize=21)
        plt.yticks(self.param_storer.y_ticks_range)
        axes.set_ylim(self.param_storer.y_lim_range)

        tick_size = 25
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_size)
        for tick in axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_size)

            # margins on x and y side
        axes.margins(x=0.035)
        axes.margins(y=0.035)

    # plot a method and return a legend handle
    def __plot_one_method(self, i, result, plot_bars):
        name = ParseName(result[0])

        rewards = result[1]
        error_bars = result[2]

        # hack for EI
        single_point_methods = len(rewards) == self.total_budget + 1
        adjusted_time_steps = range(self.total_budget + 1) if single_point_methods else self.samples_collected
        marker_size = 9 if single_point_methods else 18

        if plot_bars:
            marker_size = 5

        line_style = '-' if i < 8 else '--'
        marker_index = i

        # dirty hack to make it unfilled
        plt.plot(adjusted_time_steps, rewards, lw=1.0, linestyle=line_style, marker=self.markers[marker_index],
                 markersize=marker_size,
                 markerfacecolor="None",
                 markeredgewidth=2, markeredgecolor=self.color_sequence[i], color=self.color_sequence[i])

        if plot_bars:
        # if plot_bars and error_bars:

            plt.errorbar(adjusted_time_steps, rewards, yerr=error_bars, color=self.color_sequence[i], lw=0.1)

        patch = mlines.Line2D([], [], linestyle=line_style, color=self.color_sequence[i],
                              marker=self.markers[marker_index],
                              markerfacecolor="None",
                              markeredgewidth=3, markeredgecolor=self.color_sequence[i],
                              markersize=10, label=name)

        return patch

    def _get_param_storer(self):
        if self.dataset_type == DatasetEnum.Simulated:
            return SimulatedDatasetPlotParamStorer(self.plotting_type)
        elif self.dataset_type == DatasetEnum.Road:
            return RoadDatasetPlotParamStorer(self.plotting_type)
        elif self.dataset_type == DatasetEnum.Robot:
            return RobotDatasetPlotParamStorer(self.plotting_type)
        else:
            raise Exception("Unknown dataset type")


def ParseName(method_name):
    method_items = method_name.split()
    if method_items[0] == 'beta':
        number = float(method_items[2])
        method_name = r'$\beta = {}$'.format(number)
    return method_name
