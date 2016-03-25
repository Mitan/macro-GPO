import numpy as np
import sys
from DatasetInfo import DropWaveInfo, AckleyInfo, CosinesInfo, BraninInfo, GriewankInfo, McCormickInfo, \
    SixCamelInfo, HolderTableInfo
from TreePlanTester import TestScenario


# Random initial locations
def GenerateInitialLocation(current_f, batch_size):
    # todo
    # note: hardcoded
    default_grid_size_x = 20
    default_grid_size_y = 20

    domain = current_f.domain
    gap = current_f.grid_gap
    min_x = domain[0][0]
    min_y = domain[1][0]
    int_x = np.random.random_integers(low=0, high=default_grid_size_x, size=batch_size)
    int_y = np.random.random_integers(low=0, high=default_grid_size_y, size=batch_size)

    # coordinates of location by x and by y
    int_x = min_x + gap * int_x
    int_y = min_y + gap * int_y

    return np.asarray([[int_x[i], int_y[i]] for i in range(batch_size)])


def GetBeta(iteration):
    beta_values = [0.0, 0.5, 1.0,  3.0, 5.0, 10.0]
    return beta_values[iteration]

def GetSimulatedFunction(i):
    if i == 0:
        return AckleyInfo()
    elif i ==1:
        return DropWaveInfo()
    elif i ==2:
        return CosinesInfo()

    elif i==3:
        return BraninInfo()
    elif i==4:
        return GriewankInfo()
    elif i==5:
        return McCormickInfo()
    elif i==6:
        return SixCamelInfo()
    else:
        return HolderTableInfo()

if __name__ == '__main__':

    args = sys.argv
    batch_size = 2
    save_trunk = "./tests/"

    # should be passed as params
    function_iteration = int(args[1])
    beta_iteration = int(args[2])
    location_iteration = int(args[3])

    #function_iteration = 2
    #beta_iteration = 1
    #location_iteration = 2
    current_function = GetSimulatedFunction(function_iteration)
    initial_location = GenerateInitialLocation(current_function, batch_size)
    beta = GetBeta(beta_iteration)

    print "function is " + str(current_function.name)
    print "beta is " + str(beta)
    print "location " + str(location_iteration) +  " is "  + str(initial_location)

    TestScenario(b=batch_size, beta=beta, locations=initial_location, i = location_iteration,  simulated_func=current_function,
                 save_trunk=save_trunk)
