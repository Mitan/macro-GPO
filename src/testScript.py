from StringIO import StringIO
import numpy as np
from src.DatasetUtils import GenerateModelFromFile


root_path = '../tests/b4_sAD_loc0_h3/seed71/'
method = 'qEI'

def GetGCoefficient(root_folder, method_name):
    summary_path = root_folder + method_name + '/summary.txt'
    dataset_path = root_folder + 'dataset.txt'
    # dateset_path = root_folder + 'dataset.txt'
    lines = open(summary_path).readlines()

    first_line_index = 28
    # todo NB
    # a hack
    # for some cases strangely numbers are written to file in scientific format, then measurements
    # occupy different number of lines
    last_line_index = 31 if lines[31].strip()[-1] == ']' else 33

    stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index+1])
    joined_lines = " ".join(stripped_lines)[1:-1]
    a = StringIO(joined_lines)

    # all measurements obtained by the robot
    measurements = np.genfromtxt(a)

    #    assert that we have 21 measurement (1 initial + 4 * 5)
    assert measurements.shape[0] == 21
    # assert we parsed them all as numbers
    assert not np.isnan(measurements).any()

    initial_measurement = measurements[0]
    max_found = max(measurements)
    model = GenerateModelFromFile(dataset_path)
    true_max = model.GetMax()
    G = (max_found - initial_measurement) / (true_max - initial_measurement)
    return G

# print GetGCoefficient(root_path, method)

filename = './taxi18.dom'
lines = open(filename).readlines()
number_of_points = len(lines)

locs = np.empty((number_of_points, 2))
neighbours = np.empty((number_of_points, 9))
vals = np.empty((number_of_points, ))

for i, line in enumerate(lines):
    l = line
    a = StringIO(l)
    current_point = np.genfromtxt(a)

    current_neighbours = current_point[4:]
    neighbours_len = len(current_neighbours)
    assert neighbours_len < 10

    np.copyto(locs[i, :], current_point[0:2])
    np.copyto(neighbours[i, :neighbours_len], current_neighbours)
    vals[0] = current_point[2]

"""
data = np.genfromtxt(filename)
print data.shape
locs = data[:, :2]
vals = data[:, 2]
neighbours = data[:, 4:]
"""