import matplotlib

# Force matplotlib to not use any Xwindows backend.
from src.PlottingEnum import PlottingMethods

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


def PlotData(results, dataset, output_file_name, plottingType):
    if not results:
        return

    # 0 is width, 1 is height
    plt.rcParams["figure.figsize"] = [6, 9]

    color_sequence = ['red', 'green', 'blue', '#e377c2', '#17becf', 'orange',
                      '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#7f7f7f',
                      '#8c564b', '#c49c94', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', 'yellow']

    markers = ["o", "v", "^", "s", "*", "1", "2", "3"]

    # include first step before planning
    number_of_steps = len((results[0])[1])
    batch_size = 20 / (number_of_steps - 1)

    # legend_loc = 2 if isTotalReward else 1

    # time_steps = range(number_of_steps)
    # show samples obtained instead
    time_steps = [i * batch_size for i in range(number_of_steps)]
    # for legends
    handles = []
    for i, result in enumerate(results):
        name = ParseName(result[0])

        rewards = result[1]

        # hack for EI
        adjusted_time_steps = range(21) if (name == 'EI (all MA)' or name == 'PI') else time_steps

        # previous version with small filled markers
        # plt.plot(t, rewards, lw=1.0, color=color_sequence[i],  marker=markers[i])

        marker_index = i if i < 8 else 0

        # dirty hack to make it unfilled
        plt.plot(adjusted_time_steps, rewards, lw=1.0, marker=markers[marker_index], markersize=15,
                 markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=color_sequence[i], color=color_sequence[i])

        # patch = mpatches.Patch(color=color_sequence[i], label=name)

        patch = mlines.Line2D([], [], color=color_sequence[i], marker=markers[marker_index], markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=color_sequence[i], markersize=10, label=name)

        handles.append(patch)

    plt.xticks(time_steps)
    plt.xlabel("No. of samples collected")
    axes = plt.axes()
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if dataset == 'simulated':
        if plottingType == PlottingMethods.TotalReward:
            plt.ylabel("Total Rewards")
            plt.yticks(range(-4, 13))
            axes.set_ylim([-3, 11])
            legend_loc = 2
        elif plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total Rewards")
            plt.yticks(range(-5, 14))
            # plt.axis((x1, x2, -4, 12))
            axes.set_ylim([-5, 12])
            legend_loc = 2
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret")
            plt.yticks(np.arange(1.4, 3.2, 0.2))
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            # axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 10))
            # plt.yticks(np.arange(0, 10 ** 8, 10 ** 7))
            legend_loc = 1
        else:
            raise
    elif dataset == 'road':
        if plottingType == PlottingMethods.TotalReward:
            plt.ylabel("Total Rewards")
            plt.yticks(range(-1, 8))
            legend_loc = 2
        elif plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total Rewards")
            axes.set_ylim([-1, 8])
            plt.yticks(range(-1, 9))
            legend_loc = 2
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret")
            plt.yticks(np.arange(1.5, 4, 0.5))
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            legend_loc = 1
        else:
            raise
    elif dataset == 'robot':
        if plottingType == PlottingMethods.TotalReward:
            plt.ylabel("Total Rewards")
            plt.yticks(range(-1, 16))
            legend_loc = 2
        elif plottingType == PlottingMethods.TotalRewardBeta:
            plt.ylabel("Total Rewards")
            axes.set_ylim([-1, 15])
            plt.yticks(range(-1, 16))
            legend_loc = 2
        elif plottingType == PlottingMethods.SimpleRegret:
            plt.ylabel("Simple regret")
            legend_loc = 1
        elif plottingType == PlottingMethods.Nodes:
            plt.ylabel("No. of nodes expanded")
            axes.set_yscale('log')
            legend_loc = 1
        else:
            raise

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

    plt.legend(handles=handles, loc=legend_loc)
    # plt.savefig(folder_name + file_name)

    # margins on x and y side
    axes.margins(x=0.02)

    # if not isTotalReward:
    axes.margins(y=0.02)

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
