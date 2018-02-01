import sys
import os

import GPyOpt
import numpy as np

# from TestScenario import *
from src.DatasetUtils import GenerateRobotModelFromFile
from BBOVisualize import drawPlot


def BOLoop(start_location, fake_location, domain, f):
    X_init = np.vstack((start_location, fake_location))
    # Y_init = np.array([func_model(start_location)[0], func_model(fake_location)[0]])
    Y_init = np.vstack((f([start_location]), f([fake_location])))
    # print Y_init
    myBopt = GPyOpt.methods.BayesianOptimization(f=f,  # function to optimize
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
                                                 de_duplication=True,
                                                 maximize=True)
    max_iter = 4  # evaluation budget
    myBopt.run_optimization(max_iter)
    myBopt._print_convergence()
    print myBopt.X.shape
    # print myBopt.x_opt, func_model([myBopt.x_opt])
    return myBopt.X


def PerformBOForOneSeed(seed, m, my_save_folder_root, batch_size):

    def func_model(location):
        return np.array([[m(location[0])]])

    save_folder = my_save_folder_root + "seed" + str(seed) + "/"
    start_location = m.LoadRandomLocation(save_folder)

    domain = [{'name': 'locations', 'type': 'bandit', 'domain': m.locations}]
    X_ans = np.zeros((2, 0))

    while X_ans.shape[0] != 22:
        # doesn't work with only one starting location
        fake_location = m.GetRandomStartLocation(batch_size=batch_size)

        if np.array_equal(start_location, fake_location):
            fake_location = m.GetRandomStartLocation(batch_size=batch_size)

        X_ans = BOLoop(start_location=start_location, fake_location=fake_location,
                       domain=domain, f=func_model)

    # delete fake point
    # X_ans = np.delete(X_ans, 1, 0)
    """
    for i in range(21):
        print func_model(X_ans[i:i + 1, :])
    """
    Y_ans = np.array([model(X_ans[i, :]) for i in range(X_ans.shape[0])])
    print X_ans
    print X_ans.shape
    print Y_ans

    Visualize_LLP(found_locations=X_ans, found_values=Y_ans,
                  save_folder=save_folder, model=m, batch_size=batch_size)


def Visualize_LLP(found_locations, found_values, save_folder, model, batch_size):
    method_folder = save_folder + 'bbo-llp/'
    try:
        os.makedirs(method_folder)
    except OSError:
        if not os.path.isdir(method_folder):
            raise

    np.savetxt(X=found_locations, fmt="%.2f", fname=method_folder + 'found_locations.txt')
    np.savetxt(X=found_values, fmt="%.4f", fname=method_folder + 'rewards.txt')

    time_steps = 20 / batch_size
    for i in range(time_steps):
        drawPlot(all_locations=model.locations, values=model.values,
                 path_points=found_locations, save_path=method_folder, current_step=i, batch_size=batch_size)


if __name__ == '__main__':
    args = sys.argv

    #seed_0 = int(args[1])
    # seed_0 = 1

    time_slot = 16

    t, batch_size = (4, 5)

    my_save_folder_root = "../noise_robot_tests/release/all_tests_release/"

    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(
        time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'
    model = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                       neighbours_filename=neighbours_file)
    for seed_0 in range(35):
        PerformBOForOneSeed(seed=seed_0, m=model, my_save_folder_root=my_save_folder_root, batch_size=batch_size)
