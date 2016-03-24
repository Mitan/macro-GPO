import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#for each result, result[0] is the name, result[1] is the data as list of rewards
def PlotData(number_of_steps, results):
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    # evenly sampled time at 200ms intervals
    t = range(number_of_steps+1)
    # for legends
    handles = []
    for i, result in enumerate(results):

        name = result[0]
        # add zero at first step
        assert len(result[1]) == number_of_steps
        rewards = [0.0] + result[1]
        plt.plot(t, rewards,lw=1.0, color=color_sequence[i])
        patch = mpatches.Patch(color=color_sequence[i], label=name)
        handles.append(patch)
    plt.legend(handles=handles, loc = 2)
    path_to_file = 'test.png'
    plt.savefig(path_to_file , bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    first = ['lev1', [1,2,3,4,5]]
    second = ['lev2', [1,2,3,4,6]]
    third = ['lev3', [1,2,3,5,10]]
    results = [first, second, third]
    PlotData(5, results)