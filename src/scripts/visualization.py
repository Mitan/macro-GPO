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
        loc = (numbers[1], numbers[0])
        locations.append(loc)
    assert len(locations) == 21
    return locations
    # return list(reversed(locations))


def MapStaticPlot(grid_extent, ground_truth, path_points, save_path, xy_min, window_size):
    #grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],
    #                grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

    mmax = -10 ** 10
    mmin = 10 ** 10
    for q in ground_truth:
        # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        if q is not None:
            mmax = max(np.amax(np.amax(q)), mmax)
            mmin = min(np.amin(np.amin(q)), mmin)

    # fig_size = (6, 11)
    fig_size = (12, 8)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    # axes = plt.axes()
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    im = ax.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent, origin='lower',
                     # cmap='Greys', vmin=mmin, vmax=mmax)
                     vmin=mmin, vmax=mmax)

    # todo note hardcoded
    time_steps = 20

    # time_steps point + initial one
    for i in range(time_steps + 1):
        # for initial point and ends of macro-actions (visited locations)
        if i % batch_size == 0:
            circle_patch = plt.Circle(path_points[i], 0.2, color='black')
            ax.add_patch(circle_patch)

        # las location doesn't have an arrow
        if i == time_steps:
            pass

        else:
            x_0, y_0 = path_points[i]
            x_1, y_1 = path_points[i + 1]
            patch = plt.Arrow(x_0, y_0, x_1 - x_0, y_1 - y_0, color='black')
            ax.add_patch(patch)

    patch = plt.Rectangle(xy=xy_min, width=window_size, height=window_size, edgecolor='black', fill=False, linewidth=4)
    ax.add_patch(patch)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()


def MapAnimatedPlot(grid_extent, ground_truth, path_points, save_path):
    # grid_extent2 = [grid_extent[0], grid_extent[1], grid_extent[3],grid_extent[2]]  # Swap direction of grids in the display so that 0,0 is the top left

    mmax = -10 ** 10
    mmin = 10 ** 10
    for q in ground_truth:
        # for q in [ground_truth, posterior_mean_before, posterior_mean_after]:
        if q is not None:
            mmax = max(np.amax(np.amax(q)), mmax)
            mmin = min(np.amin(np.amin(q)), mmin)

    fig_size = (6, 8)
    fig = plt.figure(figsize=fig_size)
    axes = fig.add_subplot(111)
    # axes = plt.axes()
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    im = axes.imshow(ground_truth, interpolation='nearest', aspect='auto', extent=grid_extent,origin='lower',
                     # cmap='Greys', vmin=mmin, vmax=mmax)
                     vmin=mmin, vmax=mmax, animated=True)

    # todo note hardcoded
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
        circle_patch = plt.Circle(locations[mod_i], 0.2, color='white')
        ax.add_patch(circle_patch)

    # last stage for clearing
    if mod_i == animation_stages - 1:
        # arrows + circles at visited locations + initial point
        num_patches = time_steps + time_steps / batch_size + 1
        ax.patches = ax.patches[:-num_patches]

    # las location doesn't have an arrow
    elif mod_i == animation_stages - 2:
        pass

    else:
        x_0, y_0 = locations[mod_i]
        x_1, y_1 = locations[mod_i + 1]
        patch = plt.Arrow(x_0, y_0, x_1 - x_0, y_1 - y_0, color='black')
        ax.add_patch(patch)

    iteration += 1
    return im,


if __name__ == "__main__":
    hyper_storer = RoadHypersStorer_Log18()
    t, batch_size, num_samples = (4, 5, 250)
    time_slot = 18

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'
    m = GenerateRoadModelFromFile(filename)

    # example
    X_crop = (70, 18)
    Y_crop = (15, 23)
    input_folder = '../../mmmle_h1/'
    # the size of the cropping window
    cropping_constant = 12

    #todo note need to reverse
    X_crop = (30, 50)
    Y_crop = (0, 30)
    input_folder = '../../o/normal_28pe/'
    cropping_constant = 20

    X_crop = (30, 50)
    Y_crop = (12, 18)
    input_folder = '../../o/normal_7_qEI/'
    cropping_constant = 20

    X_crop = (23, 57)
    Y_crop = (13, 17)
    input_folder = '../../o/beta3_1_beta0.05/'
    cropping_constant = 20

    locs = GetAllLocations(input_folder + 'summary.txt', batch_size)

    ground_truth = np.vectorize(lambda x, y: m([y, x]))

    X_min = hyper_storer.grid_domain[1][0]
    X_max = hyper_storer.grid_domain[1][1]
    Y_min = hyper_storer.grid_domain[0][0]
    Y_max = hyper_storer.grid_domain[0][1]

    XGrid = np.arange(X_min, X_max - 1e-10, hyper_storer.grid_gap)
    YGrid = np.arange(Y_min, Y_max - 1e-10, hyper_storer.grid_gap)
    XGrid, YGrid = np.meshgrid(XGrid, YGrid)

    cropped_X_min = X_min + X_crop[0]
    cropped_X_max = X_max - X_crop[1]
    cropped_Y_min = Y_min + Y_crop[0]
    cropped_Y_max = Y_max - Y_crop[1]

    cropped_XGrid = np.arange(cropped_X_min, cropped_X_max - 1e-10, hyper_storer.grid_gap)
    cropped_YGrid = np.arange(cropped_Y_min, cropped_Y_max - 1e-10, hyper_storer.grid_gap)
    cropped_XGrid, cropped_YGrid = np.meshgrid(cropped_XGrid, cropped_YGrid)

    # full image
    MapStaticPlot(
        grid_extent=[X_min, X_max, Y_min, Y_max],
        ground_truth=ground_truth(XGrid, YGrid),
        path_points=locs,
        save_path=input_folder + 'ani_summary_full.png',
        xy_min=(cropped_X_min, cropped_Y_min),
        window_size=cropping_constant)

    # zoomed image
    MapAnimatedPlot(
        grid_extent=[cropped_X_min, cropped_X_max, cropped_Y_min,cropped_Y_max],
        ground_truth=ground_truth(cropped_XGrid, cropped_YGrid),
        path_points=locs,
        save_path=input_folder + 'ani_summary_zoomed.gif')

    path_points = [np.array([[19., 75.]]),
                   np.array([[20., 75.], [21., 76.], [22., 76.], [23., 76.], [23., 77.]]),
                   np.array([[24., 77.], [25., 76.], [25., 75.], [25., 74.], [24., 73.]]),
                   np.array([[23., 73.], [22., 73.], [22., 72.], [21., 72.], [20., 72.]]),
                   np.array([[19., 72.], [18., 72.], [18., 71.], [17., 72.], [16., 73.]])]
