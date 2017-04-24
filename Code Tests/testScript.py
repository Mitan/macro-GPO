from StringIO import StringIO
import numpy as np
from scipy.spatial import distance
import re
import matplotlib.pyplot as plt


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
# plt.plot(*zip(*locs), marker='o', color='r', ls='')
# plt.show()


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