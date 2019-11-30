import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
from matplotlib.lines import Line2D

from src.enum.DatasetEnum import DatasetEnum

mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def plot_graph(data, root_path, output_file_name):
    axes = plt.axes()
    # axes.margins(x=0.035)
    # axes.margins(y=0.035)

    plt.xscale("log")

    plt.xlabel("Avg. time per stage (seconds)", fontsize=20)
    plt.ylabel("Simple regret after 20 observations", fontsize=20)

    axes.set_ylim((1.31, 1.62))

    # axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # points
    for k, v in data.items():
        if k == '$q$-EI':
            continue

        if k.split()[0] == '$\epsilon$-M-GPO':
            dot_color = 'ro'
            text_color = 'red'
        else:
            dot_color = 'bo'
            text_color = 'blue'

        plt.plot(v[0], v[2], dot_color)
        annotate(axes=axes,
                 k=k,
                 text_color=text_color,
                 data=data)

    draw_lines(data, axes)

    # plt.savefig(root_path + output_file_name, format='eps', dpi=1000, bbox_inches='tight')
    plt.savefig(root_path + output_file_name)
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
    v1 = data['$\epsilon$-M-GPO  $H = 3$ $N = 5$']
    v2 = data['$\epsilon$-M-GPO  $H = 3$ $N = 30$']
    v3 = data['$\epsilon$-M-GPO  $H = 3$ $N = 100$']

    x = [v1[0], v2[0], v3[0]]
    index = 2
    y = [v1[index], v2[index], v3[index]]
    line = Line2D(x, y, color='red', linewidth=2.0)
    axes.add_line(line)


if __name__ == "__main__":
    root_path = '../../tests/simulated_h3_b4/'
    input_file = root_path + 'time_perf_data.txt'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        split = line.split(',')
        data[split[0]] = map(float, split[1:])

    plot_graph(data=data,
               root_path=root_path,
               output_file_name='regrets_time.eps')
