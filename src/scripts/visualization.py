from StringIO import StringIO

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
    vis = Vis2d()

    vis.MapAnimatedPlot(
        grid_extent=[hyper_storer.grid_domain[0][0], hyper_storer.grid_domain[0][1], hyper_storer.grid_domain[1][0],
                     hyper_storer.grid_domain[1][1]],
        ground_truth=ground_truth(XGrid, YGrid),
        path_points=[np.array([[19., 75.]]),
                     np.array([[20., 75.], [21., 76.], [22., 76.], [23., 76.], [23., 77.]]),
                     np.array([[24., 77.], [25., 76.], [25., 75.], [25., 74.], [24., 73.]]),
                     np.array([[23., 73.], [22., 73.], [22., 72.], [21., 72.], [20., 72.]]),
                     np.array([[19., 72.], [18., 72.], [18., 71.], [17., 72.], [16., 73.]])],
        save_path=input_folder +'ani_summary.png')




