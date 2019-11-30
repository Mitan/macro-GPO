import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
from matplotlib.lines import Line2D

mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib.lines as mlines

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
mpl.rcParams['legend.numpoints'] = 1


def plot_graph(data, root_path, output_file_name):
    color_sequence = ['#e41a1c', '#f781bf', 'blue', '#377eb8', '#4daf4a', '#984ea3',
                      '#ff7f00', 'black', '#a65628', ]

    markers = [ "v", "^", "o", "s", "*", "1", "2", "x", "|"]

    axes = plt.axes()
    # axes.margins(x=0.035)
    # axes.margins(y=0.035)

    plt.xscale("log")

    plt.xlabel("Avg. time per stage (seconds)", fontsize=20)
    plt.ylabel("Simple regret after 20 observations", fontsize=20)

    axes.set_ylim((1.15, 1.62))

    # axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    handles = []
    names = []

    draw_lines(data, axes)
    # points
    for i, v in enumerate(data):
        # handle = self.__plot_one_method(i, result, plot_bars)
        # handles.append(handle)

        marker_index = i
        color = color_sequence[0] if i in range(3) else color_sequence[i]
        point, = plt.plot(v[1][0], v[1][2], color=color,
                 marker=markers[marker_index],
                 markersize=15,
                 markeredgewidth=5,
                 markerfacecolor=color,
                 markeredgecolor=color)

        # handles.append(point)
        # names.append(v[0])
        patch = mlines.Line2D([], [], color=color,
                              marker=markers[marker_index],
                              markerfacecolor=color,
                              markeredgewidth=3, markeredgecolor=color,
                              markersize=10, label=v[0], linestyle="None")

        handles.append(patch)

    plt.legend(handles=handles, loc=1, prop={'size': 12})
    # plt.legend(handles, names, loc=1, prop={'size': 12})
    # plt.legend(handles, names, loc=1, prop={'size': 12})

    plt.savefig(root_path + output_file_name, format='eps', dpi=1000, bbox_inches='tight')
    # plt.savefig(root_path + output_file_name)
    plt.clf()
    plt.close()


def annotate(axes, k, data, text_color):
    if k == '$\epsilon$-M-GPO  $H = 3$ $N = 100$':
        axes.annotate(k, (data[k][0], data[k][2]), color=text_color, xytext=(0.5, 1.35),
                      arrowprops=dict(facecolor='black', linewidth=1, arrowstyle='->'))
    elif k == '$\epsilon$-M-GPO  $H = 3$ $N = 30$':
        axes.annotate(k, (data[k][0], data[k][2]), color=text_color, xytext=(8.0, 1.395),
                      arrowprops=dict(facecolor='black', linewidth=1, arrowstyle='->'))

    elif k == '$\epsilon$-M-GPO  $H = 3$ $N = 5$':
        axes.annotate(k, (data[k][0], data[k][2]), color=text_color, xytext=(0.2, 1.45),
                      arrowprops=dict(facecolor='black', linewidth=1, arrowstyle='->'))
    else:
        axes.annotate(k, (data[k][0], data[k][2]), color=text_color,
                      xytext=(4, 0), textcoords="offset points")


def draw_lines(data, axes):
    # v1 = data['$\epsilon$-M-GPO  $H = 4$ $N = 5$']
    # v2 = data['$\epsilon$-M-GPO  $H = 4$ $N = 30$']
    # v3 = data['$\epsilon$-M-GPO  $H = 4$ $N = 50$']

    v1 = data[0][1]
    v2 = data[1][1]
    v3 = data[2][1]

    x = [v1[0], v2[0], v3[0]]
    index = 2
    y = [v1[index], v2[index], v3[index]]
    line = Line2D(x, y, color='red', linewidth=2.0)
    axes.add_line(line)


if __name__ == "__main__":
    root_path = '../../tests/simulated_h3_b4/'
    root_path = '../../tests/simulated_h4/'
    input_file = root_path + 'man_time_perf_data.txt'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        split = line.split(',')
        data.append([split[0], map(float, split[1:])])

    plot_graph(data=data,
               root_path=root_path,
               output_file_name='regrets_time.eps')
