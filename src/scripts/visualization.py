import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile
from src.HypersStorer import RoadHypersStorer_Log18
from src.Vis2d import Vis2d


def GetAllLocations(summary_filename, batch_size):

    lines = open(summary_filename).readlines()
    first_line_index = 1 + batch_size + 1
    last_line_index = first_line_index + 1 + 20
    stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index])
    locations = []
    for str_line in stripped_lines:
        string_numbers = str_line.replace('[', ' ').replace(']', ' ').split()
        numbers = map(float, string_numbers)
        loc = (numbers[0], numbers[1])
        locations.append(loc)
    assert len(locations) == 21
    return locations


def MapAnimatedPlot(grid_extent, ground_truth,path_points,save_path):

    grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                    grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

    mmax = -10 ** 10
    mmin = 10 ** 10
    for q in ground_truth:
        # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        if q is not None:
            mmax = max(np.amax(np.amax(q)), mmax)
            mmin = min(np.amin(np.amin(q)), mmin)

    fig = plt.figure(figsize=(8, 11))
    axes = fig.add_subplot(111)
    # axes = plt.axes()
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                     # cmap='Greys', vmin=mmin, vmax=mmax)
                     vmin=mmin, vmax=mmax, animated=True)
    """
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    y = [50, 10, 40, 50, 70, 20, 15, 60, 20]
    batch_size = 2
    time_steps = 8
    i = 0

    # initila point + time_steps points + one for clearing
    animation_stages = 1 + time_steps + 1
    """

    if not path_points == None and path_points:
        # batch size
        # path points is a list
        number_of_points = len(path_points)

        for i in xrange(1, number_of_points):
            # both are batches of points
            prev = path_points[i - 1]
            current = path_points[i]

            prev_end = prev[-1, :]
            current_start = current[0, :]

            """
            axes.arrow(prev_end[0], prev_end[1],
                       current_start[0] - prev_end[0],
                       current_start[1] - prev_end[1], edgecolor='black')
            """
            x_coord = (prev_end[0], current_start[0])
            y_coord = (prev_end[1], current_start[1])
            axes.plot(x_coord, y_coord, color='black', linewidth=3)

            # here we need to draw k - 1 arrows
            # coz in total there will be k and the first on is already drawn

            # k should always be equal to batch_size though
            k = current.shape[0]

            for j in xrange(0, k - 1):
                # both a locations [x,y]
                current_point = current[j, :]
                next_point = current[j + 1, :]
                """
                axes.arrow(current_point[0], current_point[1],
                           next_point[0] - current_point[0],
                           next_point[1] - current_point[1], edgecolor='black')
                """
                x_coord = (current_point[0], next_point[0])
                y_coord = (current_point[1], next_point[1])
                axes.plot(x_coord, y_coord, color='black', linewidth=3)

    plt.savefig(save_path)

    plt.show()

    plt.clf()
    plt.close()


def updatefig(*args):
    global i

    mod_i = i % (animation_stages)
    # for initial point and ends of macro-actions (visited locations)
    if mod_i % batch_size == 0:
        circle_patch = plt.Circle((x[mod_i], y[mod_i]), 2, color='black')
        ax.add_patch(circle_patch)
        plt.pause(1.5)

    # last stage for clearing
    if mod_i == animation_stages - 1:
        plt.pause(2.0)
        # arrows + circles at visited locations + initial point
        num_patches = time_steps + time_steps/batch_size + 1
        ax.patches = ax.patches[:-num_patches]

    # las location doesn't have an arrow
    elif mod_i == animation_stages - 2:
        pass

    else:
        patch = plt.Arrow(x[mod_i] , y[mod_i] , x[mod_i+1] - x[mod_i], y[mod_i+1] - y[mod_i], color='black')
        ax.add_patch(patch)
        plt.pause(1.0)

    i+= 1

    return im,

if __name__ == "__main__":

    hyper_storer = RoadHypersStorer_Log18()
    t, batch_size, num_samples = (4, 5, 250)
    time_slot = 18

    input_folder = '../../mmmle_h1/'
    locations = GetAllLocations(input_folder + 'summary.txt', batch_size)

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'
    m = GenerateRoadModelFromFile(filename)

    XGrid = np.arange(hyper_storer.grid_domain[0][0], hyper_storer.grid_domain[0][1] - 1e-10, hyper_storer.grid_gap)
    YGrid = np.arange(hyper_storer.grid_domain[1][0], hyper_storer.grid_domain[1][1] - 1e-10, hyper_storer.grid_gap)
    XGrid, YGrid = np.meshgrid(XGrid, YGrid)

    ground_truth = np.vectorize(lambda x, y: m([x, y]))

    # Plot graph of locations
    MapAnimatedPlot(
        grid_extent=[hyper_storer.grid_domain[0][0], hyper_storer.grid_domain[0][1], hyper_storer.grid_domain[1][0],
                     hyper_storer.grid_domain[1][1]],
        ground_truth=ground_truth(XGrid, YGrid),
        path_points=[np.array([[19., 75.]]),
                   np.array([[20., 75.], [21., 76.], [22., 76.], [23., 76.], [23., 77.]]),
                   np.array([[24., 77.], [25., 76.], [25., 75.], [25., 74.], [24., 73.]]),
                   np.array([[23., 73.], [22., 73.], [22., 72.], [21., 72.], [20., 72.]]),
                   np.array([[19., 72.], [18., 72.], [18., 71.], [17., 72.], [16., 73.]])],
        save_path=input_folder +'ani_summary.png')

    path_points = [np.array([[19., 75.]]),
                   np.array([[20., 75.], [21., 76.], [22., 76.], [23., 76.], [23., 77.]]),
                   np.array([[24., 77.], [25., 76.], [25., 75.], [25., 74.], [24., 73.]]),
                   np.array([[23., 73.], [22., 73.], [22., 72.], [21., 72.], [20., 72.]]),
                   np.array([[19., 72.], [18., 72.], [18., 71.], [17., 72.], [16., 73.]])]




