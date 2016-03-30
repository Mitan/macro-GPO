import sys

import numpy as np
from TestScenario import TestScenario
from ResultsPlotter import PlotData

from DatasetInfo import DropWaveInfo, AckleyInfo, CosinesInfo, BraninInfo, GriewankInfo, McCormickInfo, \
    SixCamelInfo, HolderTableInfo, Log_K_Info, Log_P_Info



# Random initial locations
def GenerateInitialLocation(current_f, batch_size):
    domain = current_f.domain
    gap = current_f.grid_gap
    min_x, max_x = domain[0]
    min_y, max_y = domain[1]

    # number of samples at each coordinate
    default_grid_size_x = float(max_x - min_x) / gap
    default_grid_size_y = float(max_y - min_y) / gap

    # todo
    # should add some check for this
    default_grid_size_x = int(default_grid_size_x)
    default_grid_size_y = int(default_grid_size_y)

    # in current implementation higher limit is not included hence -1
    int_x = np.random.random_integers(low=0, high=default_grid_size_x - 1, size=batch_size)
    int_y = np.random.random_integers(low=0, high=default_grid_size_y - 1, size=batch_size)

    # coordinates of location by x and by y
    int_x = min_x + gap * int_x
    int_y = min_y + gap * int_y

    return np.asarray([[int_x[i], int_y[i]] for i in range(batch_size)])


def GetBeta(iteration):
    beta_values = [0.0, 1.0, 3.0, 5.0, 10.0, 50.0]
    return beta_values[iteration]


def GetSimulatedFunction(i):
    if i == 0:
        return AckleyInfo()
    elif i == 1:
        return DropWaveInfo()
    elif i == 2:
        return CosinesInfo()

    elif i == 3:
        return BraninInfo()
    elif i == 4:
        return GriewankInfo()
    elif i == 5:
        return McCormickInfo()
    elif i == 6:
        return SixCamelInfo()
    elif i == 7:
        return HolderTableInfo()
    elif i == 8:
        return Log_K_Info()
    else:
        return Log_P_Info()


if __name__ == '__main__':

    args = sys.argv
    batch_size = 2

    #function_iteration = 8
    #location_iteration = 0
    #beta_iteration = 0

    """
    # should be passed as params
    function_iteration = int(args[1])
    beta_iteration = int(args[2])
    location_iteration = int(args[3])
    """

    """
    zero_locations = [np.asarray([[15., 16.], [13., 8.]]), np.asarray([[9., 12.], [18., 7.]]),
                      np.asarray([[10., 18.], [11., 15.]]), np.asarray([[8., 13.], [15., 12.]]),
                      np.asarray([[11., 11.], [16., 10.]]), np.asarray([[11., 9.], [18., 12.]])]

    half_locations = [np.asarray([[9., 13.], [10., 14.]]), np.asarray([[11., 11.], [14., 18.]]),
                      np.asarray([[10., 8.], [6., 13.]]), np.asarray([[16., 7.], [12., 16.]]),
                      np.asarray([[12., 15.], [10., 10.]]), np.asarray([[11., 11.], [16., 10.]])]
    """

    #current_function = GetSimulatedFunction(function_iteration)
    #initial_location = GenerateInitialLocation(current_function, batch_size)
    #beta = GetBeta(beta_iteration)
    save_trunk = './tests/'


    save_trunk = './testsBeta/'

    """
    my_save_folder_root = save_trunk + "batch"  + str(batch_size) + "/function" + str(current_function.name) +  "/location" + str(location_iteration) + "/beta" + str(beta) +"/"
    TestScenario(b=batch_size, beta=beta, location=initial_location, simulated_func=current_function,
                        my_save_folder_root = my_save_folder_root)
    """


    function_iteration = 8
    beta_values = [0.0, 1.0, 3.0, 5.0, 10.0, 50.0]
    current_function = GetSimulatedFunction(function_iteration)

    for location_iteration in range(1,4):
        plottin_results = []
        initial_location = GenerateInitialLocation(current_function, batch_size)
        for beta in beta_values:
            #for location_iteration in [5,4,3,2,1,0]:

                #current_function = GetSimulatedFunction(function_iteration)
                #initial_location = GenerateInitialLocation(current_function, batch_size)
                #initial_location = zero_locations[location_iteration] if beta_iteration == 0 else half_locations[location_iteration]
                #beta = GetBeta(beta_iteration)

                print "function is " + str(current_function.name)
                print "beta is " + str(beta)
                print "location " + str(location_iteration) + " is " + str(initial_location)

                my_save_folder_root = save_trunk + "batch"  + str(batch_size) + "/function" + str(current_function.name) +  "/location" + str(location_iteration) + "/beta" + str(beta) +"/"
                result = TestScenario(b=batch_size, beta=beta, location=initial_location, simulated_func=current_function,
                            my_save_folder_root = my_save_folder_root)
                plottin_results.append(['beta='+ str(beta), result])

        plotting_path = save_trunk + "batch"  + str(batch_size) + "/function" + str(current_function.name) +  "/location" + str(location_iteration) +"/"
        PlotData(plottin_results, plotting_path)

