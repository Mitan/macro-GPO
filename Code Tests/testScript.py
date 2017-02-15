from StringIO import StringIO
import numpy as np
import re

from src.DatasetUtils import GenerateModelFromFile, GenerateRoadModelFromFile
from src.GaussianProcess import MapValueDict

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

    stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index + 1])
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

"""
lines = open(filename).readlines()
number_of_points = len(lines)

locs = np.empty((number_of_points, 2))
neighbours = np.empty((number_of_points, 9))
vals = np.empty((number_of_points,))

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



a = RoadMapValueDict(locations=locs, values=vals, neighbours=neighbours)

a.locations
a.neighbours


data = np.genfromtxt(filename)
print data.shape
locs = data[:, :2]
vals = data[:, 2]
neighbours = data[:, 4:]


m = GenerateRoadModelFromFile(filename)
for i in m.locations:
    # a = m.neighbours[tuple(i)] if tuple(i) in m.neighbours.keys() else 'None'
    print i, m.GetNeighbours(i)



dict = {1: [2], 2: [3], 3: [4], 4: [5, 6, 1,2 ]}
dict = {(1,0): [(2,0)], (2,0): [(3,0), (4,0)], (3,0): [(4,0)], (4,0): [(3,0)]}




# batch_size = 3

#UGLY
# TODO change into generators
batch_road_macroactions = []
def ExpandActions(start, batch_size):
    # including the start, hence +1
    if len(start) == batch_size+1:
        # remove start state
        # batch_road_macroactions.append(start[1:])
        # return
        yield map(list, start[1:])
    else:
        current = start[-1]
        for next_node in dict[current]:
            if next_node in start:
                continue
            for state in ExpandActions(start + [next_node], batch_size):
                yield state

def GenerateRoadMacroActions(current_state, batch_size):
    ExpandActions([current_state], batch_size)
    return batch_road_macroactions
# print batch_road_macroactions


# for i in GenerateRoadMacroActions((1,0), 3):
#l = list(ExpandActions([(1,0)], 3))
# print l

m = GenerateRoadModelFromFile(filename)

locs = m.locations

count = 0
for loc in locs:
    length = len(m.GenerateRoadMacroActions(tuple(loc), 3))
    if length > 0:
        print loc, length
        count+=1
print count

a =  np.array([m.GetRandomStartLocation(3)])
print a.shape

        # m.GenerateRoadMacroActions(tuple(loc), 3)

print round(8.8333333333333339 , 2)

print np.around([0.374, 1.6400000001], decimals=2)

a = np.array([[1,2], [3,4], [5,6]])
for x in a:
    print a



m = GenerateRoadModelFromFile(filename)
past_locations = np.array([[1.0, 1.0]])
#print past_locations.shape

a =  np.array([m.GetRandomStartLocation(3)])
current_location = a[-1, :]
print current_location

print m.GenerateRoadMacroActions(current_location, 3)[0].shape



a =  - float("inf")
print a
print a + 1

a = np.asarray([1,2,3])
print a.shape
print a[0]
a[0] = 44
print a
"""
a = '0.0 29.0 [0, 1, 2, 3, 4]'




a = 0.2
a =  [1,2,3]
b = [3,4,5]
