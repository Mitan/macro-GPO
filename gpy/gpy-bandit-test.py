import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt

filename = './data/target_day_20140422.dat'
f = open(filename, 'r')
contents = f.readlines()

## Create a dictionary for the forecasted
forecast_dict = {}
for line in range(1, len(contents)):
    line_split = contents[line].split(' ')
    try:
        forecast_dict[line_split[0], line_split[1]][line_split[2]] = {'MaxT': float(line_split[3]),
                                                                      'MinT': float(line_split[4][:-1])}
    except:
        forecast_dict[line_split[0], line_split[1]] = {}
        forecast_dict[line_split[0], line_split[1]][line_split[2]] = {'MaxT': float(line_split[3]),
                                                                      'MinT': float(line_split[4][:-1])}
keys = forecast_dict.keys()
day_out = '0'  # 0-7
temp = 'MaxT'  # MaxT or MinT
temperature = [];
lat = [];
lon = []
for key in keys:
    temperature.append(float(forecast_dict[key][day_out][temp]))
    lat.append(float(key[0]))
    lon.append(float(key[1]))

## Create numpy arrays for the analyisis and remove Alaska and the islands
lon = np.array(lon)
lat = np.array(lat)
sel = np.logical_and(np.logical_and(lat > 24, lat < 51), np.logical_and(lon > -130, lon < -65))
stations_coordinates_all = np.array([lon, lat]).T
stations_maxT_all = np.array([temperature]).T
stations_coordinates = stations_coordinates_all[sel, :]
stations_maxT = stations_maxT_all[sel, :]

# Check the total number of stations.
print stations_maxT.shape[0]

plt.figure(figsize=(12, 7))
sc = plt.scatter(stations_coordinates[:, 0], stations_coordinates[:, 1], c='b', s=2, edgecolors='none')
plt.title('US weather stations', size=25)
plt.xlabel('Logitude', size=15)
plt.ylabel('Latitude', size=15)
plt.ylim((25, 50))
plt.xlim((-128, -65))


# plt.show()
#  Class that defines the function to optimize given the available locations
class max_Temp(object):
    def __init__(self, stations_coordinates, stations_maxT):
        self.stations_coordinates = stations_coordinates
        self.stations_maxT = stations_maxT

    def f(self, x):
        return np.dot(0.5 * (self.stations_coordinates == x).sum(axis=1), self.stations_maxT)[:, None]


func = max_Temp(stations_coordinates, stations_maxT)
domain = [{'name': 'stations', 'type': 'bandit', 'domain': stations_coordinates}]

from numpy.random import seed

seed(123)
X_init = np.array([[ -95.708313 ,  37.224239], [-124.201752,   41.755951]])
Y_init = np.array([func.f(X_init[0,:])[0],  func.f(X_init[1,:])[0]])
# need a func which takes a point as 1d array and returns a value as 2d array
print func.f(np.array([ -95.708313 ,  37.224239]))
"""
iter_count = 4
X_step = X_init
Y_step = Y_init

for i in range(iter_count):
    myBopt = GPyOpt.methods.BayesianOptimization(f=None,  # function to optimize
                                                 domain=domain,
                                                 X=X_init,
                                                 Y=Y_init,
                                                 initial_design_numdata=5,
                                                 acquisition_type='EI',
                                                 # exact_feval=True,
                                                 normalize_Y=True,
                                                 optimize_restarts=10,
                                                 # acquisition_weight=2,
                                                 evaluator_type='local_penalization',
                                                 batch_size=5,
                                                 num_cores=4,
                                                 de_duplication=True)
    bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X_step, Y=Y_step)
    x_next = bo_step.suggest_next_locations()
    y_next = func.f(x_next)

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))
"""
myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,  # function to optimize
                                             domain=domain,
                                             X = X_init,
                                             Y = Y_init,
                                             initial_design_numdata=5,
                                             acquisition_type='EI',
                                             # exact_feval=True,
                                             normalize_Y=True,
                                             optimize_restarts=10,
                                             # acquisition_weight=2,
                                             evaluator_type='local_penalization',
                                             batch_size=5,
                                             num_cores=4,
                                             de_duplication=True)

# Run the optimization
max_iter = 4  # evaluation budget
# myBopt.run_optimization(max_iter)

plt.figure(figsize=(15, 7))
jet = plt.cm.get_cmap('jet')
sc = plt.scatter(stations_coordinates[:, 0], stations_coordinates[:, 1], c=stations_maxT, vmin=0, vmax=35, cmap=jet,
                 s=3, edgecolors='none')
cbar = plt.colorbar(sc, shrink=1)
cbar.set_label(temp)
plt.plot(myBopt.x_opt[0], myBopt.x_opt[1], 'ko', markersize=10, label='Best found')
plt.plot(myBopt.X[:, 0], myBopt.X[:, 1], 'k.', markersize=8, label='Observed stations')
plt.plot(stations_coordinates[np.argmin(stations_maxT), 0], stations_coordinates[np.argmin(stations_maxT), 1], 'k*',
         markersize=15, label='Coldest station')
plt.legend()
plt.ylim((25, 50))
plt.xlim((-128, -65))

plt.title('Max. temperature: April, 22, 2014', size=25)
plt.xlabel('Longitude', size=15)
plt.ylabel('Latitude', size=15)
plt.text(-125, 28, 'Total stations =' + str(stations_maxT.shape[0]), size=20)
plt.text(-125, 26.5, 'Sampled stations =' + str(myBopt.X.shape[0]), size=20)
plt.show()

plt.figure(figsize=(8, 5))
xx = plt.hist(stations_maxT, bins=50)
plt.title('Distribution of max. temperatures', size=25)
plt.vlines(min(stations_maxT), 0, 1000, lw=3, label='Coldest station')
plt.vlines(myBopt.fx_opt, 0, 1000, lw=3, linestyles=u'dotted', label='Best found')
plt.legend()
plt.xlabel('Max. temperature', size=15)
plt.xlabel('Frequency', size=15)

# plt.show()
