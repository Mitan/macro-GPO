from StringIO import StringIO
import numpy as np
from scipy.spatial import distance
import re
import matplotlib.pyplot as plt
import math

from collections import Counter

"""
#plt.show()
dat = np.genfromtxt('../datasets/robot/real_loc')
# dat = dat[:, :-1]
locs =  dat.tolist()

locs = map(tuple,locs)
print len(locs)
locs = list(set(locs))
# locs = map(lambda x: np.array(x), locs)

len_locs = len(locs)
print  len_locs
plt.plot(*zip(*locs), marker='o', color='r', ls='')
plt.show()

d = 0
for loc in locs:
    for nei in locs:
        if loc == nei:
            continue
        # last coordinate is orientation
        dst = distance.euclidean(loc[:-1], nei[:-1])
        d+= dst
treshold =  d / (len_locs* (len_locs - 1)) / 15
print treshold

angle_treshold = np.pi / 4
av_len = 0
for loc in locs:
    neighbours = []
    for nei in locs:
        if loc == nei:
            continue
        # last coordinate is orientation
        dst = distance.euclidean(loc[:-1], nei[:-1])
        if dst < treshold and abs(loc[-1] - nei[-1] < angle_treshold):
            neighbours.append(nei)
    av_len += len(neighbours)

print float(av_len) / len_locs
"""

"""
print first.shape
print first
for line in lines:
    id =  line[0]
    if id in dict.keys():
        dict[id] = dict[id] + 1
    else:
        dict[id] = 1

print dict
"""

"""
def process_1():
    dict = {}
    # print "0"
    lines = np.genfromtxt('../datasets/intel-robot/data-matlab.txt')
    first = lines[:, 0].tolist()
    keys = []
    count = Counter(first)

    for key, value in sorted(count.items(), key=lambda kv: kv[1], reverse=True):
        if value > 40:
            keys.append(key)
            print key, value
    print keys

    out_file = open('../datasets/intel-robot/dataset.txt', 'w')
    for line in lines:
        if line[0] in keys:
            line = line.tolist()
            line = map(str, line)
            line = " ".join(line)
            print line
            out_file.write(str(line) + '\n')


def GetSlotForOneKey(all_lines, all_keys, index):
    selected_lines = all_lines[all_lines[:, 0] == all_keys[index]]
    num_points = selected_lines.shape[0]

    # check that ids do not repeat
    id_set = set(selected_lines[:, 1])
    if len(list(id_set)) != num_points:
        print index, len(list(id_set)), num_points

    f = open('../datasets/intel-robot/slots/slot_' + str(index) + '.txt', 'w')

    for i in range(num_points):
        id = selected_lines[i, 1]
        coords = coord_dict[id]
        t = selected_lines[i, 2]
        if math.isnan(t):
            print id, index
            continue
        f.write(str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(t) + '\n')
    f.close()

# dict id -> coordinates (x,y)
coord_dict = {}
coords = np.genfromtxt('../datasets/intel-robot/coordinates.txt')
for line in coords:
    # print line
    coord_dict[line[0]] = (line[1], line[2])

# process_1()

# selected keys with at least 40
keys = [65580.0, 75090.0, 54930.0, 55950.0, 64140.0, 66570.0, 67830.0, 392650.0, 51060.0, 52590.0,
        57360.0, 60180.0, 60960.0, 67380.0, 71550.0, 73020.0, 88590.0, 339220.0, 346270.0, 359050.0, 360940.0, 364900.0,
        377200.0, 415600.0, 416740.0]

# these keys have repeated values
bad_keys = [1664170.0, 1667790.0]

# all slots lines
new_lines = np.genfromtxt('../datasets/intel-robot/raw_dataset.txt')

for i in range(len(keys)):
    GetSlotForOneKey(all_lines=new_lines, all_keys=keys, index=i)
"""

print range(3, 1, -1)