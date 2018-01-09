import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib  import cm


locations_file = '../../datasets/robot/selected_slots/slot_16/final_slot_16.txt'
all_locations_data = np.genfromtxt(locations_file)
locations = all_locations_data[:, 1:3]
values = all_locations_data[:, -1:
         ]
true_locations_file = '../../datasets/robot/selected_slots/slot_16/slot_16.txt'
true_locations = np.genfromtxt(true_locations_file)[:, 1:3]

X = locations[:, 0]
Y = locations[:, 1]

mmax = np.amax(np.amax(values))
mmin = np.amin(np.amin(values))
axes = plt.axes()

axes.scatter(X, Y, s=30, c=values, vmin=mmin, vmax=mmax, cmap=cm.jet)


for loc in true_locations:
    circle = plt.Circle(loc, 0.8, color='black', fill=False)
    axes.add_artist(circle)

save_path = './'
# plt.savefig(save_path + "robot_dataset.png")
plt.savefig(save_path + "intel_lab_circled.eps", format='eps', dpi=1000, bbox_inches='tight')
plt.clf()
plt.close()