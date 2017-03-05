import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# for each result, result[0] is the name, result[1] is the data as list of rewards

def PlotData(results, isRoad,  output_file_name, isTotalReward):
    if not results:
        return
    color_sequence = ['red', 'green', 'blue', '#e377c2', '#17becf', 'orange',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#7f7f7f',
                      '#8c564b', '#c49c94', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', 'yellow']

    markers = ["o", "v", "^", "s", "*", "1", "2", "3"]

    # number of steps is the length of the first list of rewards
    number_of_steps = len((results[0])[1])

    # todo handle properly
    legend_loc = 2 if isTotalReward else 1

    time_steps = range(number_of_steps)
    # for legends
    handles = []
    for i, result in enumerate(results):
        name = result[0]

        # add zero at first step
        rewards = result[1]

        # previous version with small filled markers
        # plt.plot(t, rewards, lw=1.0, color=color_sequence[i],  marker=markers[i])

        # dirty hack to make it unfilled
        plt.plot(time_steps, rewards, lw=1.0, marker=markers[i], markersize=15, markerfacecolor="None",
                 markeredgewidth=1, markeredgecolor=color_sequence[i], color=color_sequence[i])

        # patch = mpatches.Patch(color=color_sequence[i], label=name)
        patch = mlines.Line2D([], [], color=color_sequence[i], marker=markers[i], markerfacecolor="None",
                              markeredgewidth=1, markeredgecolor=color_sequence[i], markersize=10, label=name)

        handles.append(patch)

    plt.xticks(range(number_of_steps + 1))

    if isTotalReward:
        if isRoad:
            plt.yticks(range(-1, 9))
        else:
            plt.yticks(range(-4, 13, 2))

    plt.legend(handles=handles, loc=legend_loc)
    # plt.savefig(folder_name + file_name)

    # margins on x and y side
    axes = plt.axes()
    axes.margins(x=0.02)

    plt.savefig(output_file_name, bbox_inches='tight')
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
