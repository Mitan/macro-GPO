import matplotlib

# Force matplotlib to not use any Xwindows backend.
from src.enum.PlottingEnum import PlottingMethods

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


# for each result, result[0] is the name, result[1] is the data as list of rewards

def ParseName(method_name):
    method_items = method_name.split()
    if method_items[0] == 'beta':
        number = float(method_items[2])
        method_name = r'$\beta = {}$'.format(number)
    return method_name


def PlotData(results, dataset, output_file_name, plottingType, plot_bars=False):
    if not results:
        return

    # 0 is width, 1 is height
    plt.rcParams["figure.figsize"] = [6, 9]

    color_sequence = [ 'green', 'blue', '#e377c2', 'red', '#17becf', 'orange',
                      '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#7f7f7f',
                      '#8c564b', '#c49c94', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', 'yellow']

    color_sequence = ['#e41a1c', '#377eb8', '#4daf4a','#984ea3' ,'#ff7f00' ,'black','#a65628','#f781bf', 'yellow']
    color_sequence = ['#e41a1c', '#377eb8', '#4daf4a','#984ea3' ,'#ff7f00' ,'black','#a65628','#f781bf', 'blue']

    markers = ["o", "v", "^", "s", "*", "1", "2", "x", "|"]

    # include first step before planning
    number_of_steps = len((results[0])[1])
    batch_size = 20 / (number_of_steps - 1)

    # legend_loc = 2 if isTotalReward else 1

    # time_steps = range(number_of_steps)
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
        """
        single_point_methods = name == 'EI (all)' or \
                               name == 'EI' or\
                               name == 'PI' or\
                               name == r'Rollout-$H=4 \gamma =1.0$ PI' or\
                               name == r'$H =4$ $N=20$' or\
                               name == r'$H =4$ $N=40$'
        """
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

    if dataset == 'simulated':
        if plottingType == PlottingMethods.TotalReward or plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total normalized output measurements observed by AUV", fontsize=labels_font_size)
            # plt.yticks(range(-4, 14))
            # axes.set_ylim([-3.5, 14])
            axes.set_ylim([-0.5, 20])
            plt.yticks(range(0, 20))
            legend_loc = 2
            """
            elif plottingType == PlottingMethods.TotalRewardBeta:
                plt.ylabel("Total normalized output measurements observed by UAV")
                # plt.yticks(range(-5, 12))
                # axes.set_ylim([-5, 11])
                plt.yticks(range(-4, 13))
                axes.set_ylim([-3.5, 11])
                legend_loc = 2
            """
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret", fontsize=labels_font_size)
            plt.yticks(np.arange(1.0, 3.2, 0.2))
            legend_loc = 1
        elif plottingType == PlottingMethods.CumulativeRegret:
            plt.ylabel("Average cumulative regret", fontsize=labels_font_size)
            plt.yticks(np.arange(1.0, 3.2, 0.2))
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            # axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 10))
            # plt.yticks(np.arange(0, 10 ** 8, 10 ** 7))
            legend_loc = 1
        else:
            raise Exception
    elif dataset == 'road':
        if plottingType == PlottingMethods.TotalReward or plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total normalized output measurements observed by AV", fontsize=labels_font_size)
            """
            plt.yticks(range(-1, 14))
            plt.yticks(range(0, 8))
            axes.set_ylim([-1.5, 13])
            axes.set_ylim([0, 7])
            """
            axes.set_ylim([-1.5, 6])
            plt.yticks(range(-1, 7))
            legend_loc = 2
            """
            elif plottingType == PlottingMethods.TotalRewardBeta:
                plt.ylabel("Total normalized output measurements observed by AV")
                axes.set_ylim([-1.5, 6])
                plt.yticks(range(-1, 7))
                legend_loc = 2
            """
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret", fontsize=labels_font_size)
            plt.yticks(np.arange(1.5, 4, 0.5))
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            legend_loc = 1
        else:
            raise Exception
    elif dataset == 'robot':
        if plottingType == PlottingMethods.TotalReward or plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total normalized output measurements observed by mobile robot", fontsize=labels_font_size)
            axes.set_ylim([-0.5, 14])
            plt.yticks(range(0, 15))
            legend_loc = 2
            """
            elif plottingType == PlottingMethods.TotalRewardBeta:
                plt.ylabel("Total normalized output measurements observed by mobile robot")
                axes.set_ylim([-0.5, 14])
                plt.yticks(range(0, 15))
                legend_loc = 2
            """
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret", fontsize=labels_font_size)
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            legend_loc = 1
        else:
            raise Exception

    """ 
    if isTotalReward:
        plt.ylabel("Total Rewards")
        if dataset == 'road':
            if isBeta:
                axes.set_ylim([-1, 8])
                plt.yticks(range(-1, 9))
            else:
                plt.yticks(range(-1, 8))
        elif dataset == 'robot':
            if isBeta:
                axes.set_ylim([0, 16])
                plt.yticks(range(0, 17))
            else:
                plt.yticks(range(-1, 16))
        elif dataset == 'simulated':
            if isBeta:
                plt.yticks(range(-4, 14))
                # plt.axis((x1, x2, -4, 12))
                axes.set_ylim([-4, 12])
            else:
                plt.yticks(range(-4, 13))
        else:
            raise
    else:
        plt.ylabel("Simple regret")
        if dataset == 'road':
            plt.yticks(np.arange(1.5, 4, 0.5))
        elif dataset == 'robot':
            pass
            # lt.yticks(np.arange(1.5, 4, 0.5))
        elif dataset == 'simulated':
            plt.yticks(np.arange(1.4, 3.2, 0.2))
        else:
            raise
    """

    plt.legend(handles=handles, loc=legend_loc, prop={'size': 13.5})
    # plt.savefig(folder_name + file_name)

    # margins on x and y side
    axes.margins(x=0.035)

    # if not isTotalReward:
    axes.margins(y=0.035)

    # plt.savefig(output_file_name, bbox_inches='tight')
    # plt.savefig(output_file_name, format='eps', dpi=1000)
    plt.savefig(output_file_name, format='eps', dpi=1000, bbox_inches='tight')
    # plt.savefig(folder_name + file_name, bbox_inches=1.0)

    plt.clf()
    plt.close()
    # plt.show()


if __name__ == "__main__":
    first = ['lev1', [1, 2, 3, 4, 5]]
    second = ['lev2', [1, 2, 3, 4, 6]]
    third = ['lev3', [1, 2, 3, 5, 10]]
    foru = ['lev3', [1, 4, 3, 5, 8]]
    fif = ['lev3', [1, 6, 3, 5, 11]]
    six = ['lev3', [1, 6, 3, 2, 1]]
    results = [first, second, third, foru, fif, six]
    # PlotData(results, "./")
