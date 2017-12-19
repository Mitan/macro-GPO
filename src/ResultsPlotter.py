import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


# for each result, result[0] is the name, result[1] is the data as list of rewards

def ParseName(method_name):
    method_items = method_name.split()
    if method_items[0] == 'beta':
        number = float(method_items[2])
        method_name = r'$\beta = {}$'.format(number)
    return method_name


def PlotData(results, type,  output_file_name, isTotalReward):
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
    batch_size = 20 / (number_of_steps -1)

    legend_loc = 2 if isTotalReward else 1

    # time_steps = range(number_of_steps)
    # show samples obtained instead
    time_steps = [i * batch_size for i in range(number_of_steps)]
    # for legends
    handles = []
    for i, result in enumerate(results):

        name = ParseName(result[0])

        rewards = result[1]

        # hack for EI
        adjusted_time_steps = range(21) if name=='EI' else time_steps

        # previous version with small filled markers
        # plt.plot(t, rewards, lw=1.0, color=color_sequence[i],  marker=markers[i])

        marker_index = i if i < 8 else 0

        # dirty hack to make it unfilled
        plt.plot(adjusted_time_steps, rewards, lw=1.0, marker=markers[marker_index], markersize=15, markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=color_sequence[i], color=color_sequence[i])

        # patch = mpatches.Patch(color=color_sequence[i], label=name)

        patch = mlines.Line2D([], [], color=color_sequence[i], marker=markers[marker_index], markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=color_sequence[i], markersize=10, label=name)

        handles.append(patch)

    plt.xticks(time_steps)
    plt.xlabel("No. of samples collected")

    if isTotalReward:
        plt.ylabel("Total Rewards")
        if type == 'road':
            plt.yticks(range(-1, 8))
        elif type == 'robot':
            plt.yticks(range(-1, 17))
        elif type == 'simulated':
            plt.yticks(range(-4, 13, 2))
        else:
            raise
    else:
        plt.ylabel("Simple regret")

    plt.legend(handles=handles, loc=legend_loc)
    # plt.savefig(folder_name + file_name)

    # margins on x and y side
    axes = plt.axes()
    axes.margins(x=0.02)

    if not isTotalReward:
        axes.margins(y=0.02)

    # plt.savefig(output_file_name, bbox_inches='tight')
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
