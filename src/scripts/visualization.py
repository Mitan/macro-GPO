import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile
from src.HypersStorer import RoadHypersStorer_Log18


# todo note
# for some reason the first iteration (number 0) appears two times in animation (check by printing i from update)
# dirty hack to resolve it
iteration = 0

# in 2.0 they changed default style
mpl.style.use('classic')

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

def MapStaticPlot(grid_extent, ground_truth, path_points, save_path, isZoomed):

    grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                    grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

    mmax = -10 ** 10
    mmin = 10 ** 10
    for q in ground_truth:
        # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        if q is not None:
            mmax = max(np.amax(np.amax(q)), mmax)
            mmin = min(np.amin(np.amin(q)), mmin)

    fig_size = (8,8) if isZoomed else (8, 11)
    fig = plt.figure(figsize=fig_size)
    axes = fig.add_subplot(111)
    # axes = plt.axes()
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                     # cmap='Greys', vmin=mmin, vmax=mmax)
                     vmin=mmin, vmax=mmax, animated=True)

    #todo note hardcoded
    time_steps = 20

    ani = animation.FuncAnimation(fig, updatefig, fargs=(time_steps, axes, im, path_points), interval=1000, blit=False)
    plt.show()
    ani.save(save_path, dpi=80, writer='imagemagick')


def MapAnimatedPlot(grid_extent, ground_truth, path_points, save_path, is_zoomed):

    grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
                    grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

    mmax = -10 ** 10
    mmin = 10 ** 10
    for q in ground_truth:
        # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        if q is not None:
            mmax = max(np.amax(np.amax(q)), mmax)
            mmin = min(np.amin(np.amin(q)), mmin)

    fig_size = (8,8) if is_zoomed else (8, 11)
    fig = plt.figure(figsize=fig_size)
    axes = fig.add_subplot(111)
    # axes = plt.axes()
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent2,
                     # cmap='Greys', vmin=mmin, vmax=mmax)
                     vmin=mmin, vmax=mmax, animated=True)

    #todo note hardcoded
    time_steps = 20

    ani = animation.FuncAnimation(fig, updatefig, fargs=(time_steps, axes, im, path_points), interval=1000, blit=False)
    plt.show()
    ani.save(save_path, dpi=80, writer='imagemagick')


def updatefig(i, time_steps, ax, im, locations):

    global iteration

    # initila point + time_steps points + one for clearing
    animation_stages = 1 + time_steps + 1

    mod_i = iteration % (animation_stages)
    # for initial point and ends of macro-actions (visited locations)
    if mod_i % batch_size == 0:
        circle_patch = plt.Circle(locations[mod_i], 0.2, color='black')
        ax.add_patch(circle_patch)

    # last stage for clearing
    if mod_i == animation_stages - 1:
        # arrows + circles at visited locations + initial point
        num_patches = time_steps + time_steps/batch_size + 1
        ax.patches = ax.patches[:-num_patches]

    # las location doesn't have an arrow
    elif mod_i == animation_stages - 2:
        pass

    else:
        x_0, y_0 = locations[mod_i]
        x_1, y_1 = locations[mod_i + 1]
        patch = plt.Arrow(x_0 , y_0 , x_1 - x_0, y_1 - y_0, color='black')
        ax.add_patch(patch)

    iteration+=1
    return im,

if __name__ == "__main__":

    hyper_storer = RoadHypersStorer_Log18()
    t, batch_size, num_samples = (4, 5, 250)
    time_slot = 18

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'
    m = GenerateRoadModelFromFile(filename)

    input_folder = '../../mmmle_h1/'
    locs = GetAllLocations(input_folder + 'summary.txt', batch_size)

    X_crop = (15, 23)
    Y_crop = (70, 18)

    X_min = hyper_storer.grid_domain[0][0]
    X_max = hyper_storer.grid_domain[0][1]
    Y_min = hyper_storer.grid_domain[1][0]
    Y_max = hyper_storer.grid_domain[1][1]

    X_min = X_min + X_crop[0]
    X_max = X_max - X_crop[1]
    Y_min = Y_min + Y_crop[0]
    Y_max = Y_max - Y_crop[1]

    XGrid = np.arange(X_min, X_max - 1e-10, hyper_storer.grid_gap)
    YGrid = np.arange(Y_min, Y_max - 1e-10, hyper_storer.grid_gap)
    XGrid, YGrid = np.meshgrid(XGrid, YGrid)

    ground_truth = np.vectorize(lambda x, y: m([x, y]))

    # Plot graph of locations
    MapAnimatedPlot(
        grid_extent=[X_min, X_max, Y_min,Y_max],
        ground_truth=ground_truth(XGrid, YGrid),
        path_points=locs,
        save_path=input_folder +'ani_summary_gif.gif',
        is_zoomed=True)

    path_points = [np.array([[19., 75.]]),
                   np.array([[20., 75.], [21., 76.], [22., 76.], [23., 76.], [23., 77.]]),
                   np.array([[24., 77.], [25., 76.], [25., 75.], [25., 74.], [24., 73.]]),
                   np.array([[23., 73.], [22., 73.], [22., 72.], [21., 72.], [20., 72.]]),
                   np.array([[19., 72.], [18., 72.], [18., 71.], [17., 72.], [16., 73.]])]




