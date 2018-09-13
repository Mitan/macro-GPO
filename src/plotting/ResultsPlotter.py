import matplotlib

# Force matplotlib to not use any Xwindows backend.
from src.enum.DatasetEnum import DatasetEnum
from src.enum.PlottingEnum import PlottingMethods
from src.plotting.SimulatedDatasetPlotParamStorer import SimulatedDatasetPlotParamStorer

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rc

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


class ResultGraphPlotter:

    def __init__(self, dataset_type, plotting_type, batch_size):
        self.plotting_type = plotting_type
        self.dataset_type = dataset_type
        self.param_storer = self._get_param_storer()

        # size of font at x and y label
        self.labels_font_size = 18

        # + 1 because of initial point
        plotting_num_steps = 20 / batch_size + 1

        self.samples_collected = [i * batch_size for i in range(plotting_num_steps)]

    def plot_results(self, results, plot_bars, output_file_name):
        if not results:
            return

        # 0 is width, 1 is height
        plt.rcParams["figure.figsize"] = [6, 9]
        color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 'black', '#a65628', '#f781bf', 'blue']

        markers = ["o", "v", "^", "s", "*", "1", "2", "x", "|"]

        # for legends
        handles = []

        for i, result in enumerate(results):
            name = ParseName(result[0])

            rewards = result[1]
            error_bars = result[2]

            # hack for EI
            single_point_methods = len(rewards) == 21
            adjusted_time_steps = range(21) if single_point_methods else self.samples_collected
            marker_size = 10 if single_point_methods else 20
            if plot_bars:
                marker_size = 5

            # previous version with small filled markers
            # plt.plot(t, rewards, lw=1.0, color=color_sequence[i],  marker=markers[i])

            # marker_index = i if i < 8 else 0
            linestyle = '-' if i < 8 else '--'
            marker_index = i

            # dirty hack to make it unfilled
            plt.plot(adjusted_time_steps, rewards, lw=1.0, linestyle=linestyle, marker=markers[marker_index],
                     markersize=marker_size,
                     markerfacecolor="None",
                     markeredgewidth=1, markeredgecolor=color_sequence[i], color=color_sequence[i])

            if plot_bars and error_bars:
                plt.errorbar(adjusted_time_steps, rewards, yerr=error_bars, color=color_sequence[i], lw=0.1)

            # patch = mpatches.Patch(color=color_sequence[i], label=name)

            patch = mlines.Line2D([], [], linestyle=linestyle, color=color_sequence[i], marker=markers[marker_index],
                                  markerfacecolor="None",
                                  markeredgewidth=1, markeredgecolor=color_sequence[i],
                                  markersize=10, label=name)

            handles.append(patch)
        plt.legend(handles=handles, loc=self.param_storer.legend_loc, prop={'size': 13.5})

        self.__ticks_and_margins()

        plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')
        plt.clf()
        plt.close()

    def __ticks_and_margins(self):
        plt.xticks(self.samples_collected)
        plt.xlabel("No. of observations", fontsize=self.labels_font_size)
        axes = plt.axes()
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.ylabel(self.param_storer.y_label_caption, fontsize=self.labels_font_size)
        plt.yticks(self.param_storer.y_ticks_range)
        axes.set_ylim(self.param_storer.y_lim_range)

        # margins on x and y side
        axes.margins(x=0.035)
        axes.margins(y=0.035)

    def __plot_one_method(self):
        pass

    def _get_param_storer(self):
        if self.dataset_type == DatasetEnum.Simulated:
            return SimulatedDatasetPlotParamStorer(self.plotting_type)


# for each result, result[0] is the name, result[1] is the data as list of rewards


def ParseName(method_name):
    method_items = method_name.split()
    if method_items[0] == 'beta':
        number = float(method_items[2])
        method_name = r'$\beta = {}$'.format(number)
    return method_name

"""
def PlotData(results, dataset, output_file_name, plotting_type, plot_bars=False):
    if not results:
        return

    # 0 is width, 1 is height
    plt.rcParams["figure.figsize"] = [6, 9]
    color_sequence = ['#e41a1c', '#377eb8', '#4daf4a','#984ea3' ,'#ff7f00' ,'black','#a65628','#f781bf', 'blue']

    markers = ["o", "v", "^", "s", "*", "1", "2", "x", "|"]

    # include first step before planning
    number_of_steps = len((results[0])[1])
    batch_size = 20 / (number_of_steps - 1)

     # show samples obtained instead
    time_steps = [i * batch_size for i in range(number_of_steps)]
    # for legends
    handles = []

    labels_font_size = 18
    for i, result in enumerate(results):
        name = ParseName(result[0])

        rewards = result[1]
        error_bars = result[2]

        # hack for EI
        single_point_methods = len(rewards) == 21
        adjusted_time_steps = range(21) if single_point_methods else time_steps
        marker_size = 10 if single_point_methods else 20
        if plot_bars:
            marker_size = 5


        # previous version with small filled markers
        # plt.plot(t, rewards, lw=1.0, color=color_sequence[i],  marker=markers[i])

        # marker_index = i if i < 8 else 0
        linestyle = '-' if i < 8 else '--'
        marker_index = i

        # dirty hack to make it unfilled
        plt.plot(adjusted_time_steps, rewards, lw=1.0,linestyle=linestyle, marker=markers[marker_index],
                 markersize=marker_size,
                 markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=color_sequence[i], color=color_sequence[i])

        if plot_bars and error_bars:
            plt.errorbar(adjusted_time_steps, rewards, yerr=error_bars, color=color_sequence[i],lw=0.1)

        # patch = mpatches.Patch(color=color_sequence[i], label=name)

        patch = mlines.Line2D([], [],linestyle=linestyle, color=color_sequence[i], marker=markers[marker_index],
                              markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=color_sequence[i],
                              markersize=10, label=name)

        handles.append(patch)

    plt.xticks(time_steps)
    plt.xlabel("No. of observations",fontsize=labels_font_size )
    axes = plt.axes()
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    param_storer = SimulatedDatasetPlotParamStorer(plotting_type=plotting_type)
    plt.ylabel(param_storer.y_label_caption, fontsize=labels_font_size)
    plt.yticks(param_storer.y_ticks_range)
    axes.set_ylim(param_storer.y_lim_range)

    plt.legend(handles=handles, loc=param_storer.legend_loc, prop={'size': 13.5})

    # margins on x and y side
    axes.margins(x=0.035)
    axes.margins(y=0.035)
    plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')

    plt.clf()
    plt.close()
"""