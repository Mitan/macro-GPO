import sys
import os
from random import choice
import GPyOpt
import numpy as np

# from TestScenario import *
from src.DatasetUtils import GenerateRoadModelFromFile
from BBOVisualize import drawPlot


def BOLoop1(start_location, fake_location, domain, f, batch_size):
    X_init = np.vstack((start_location, fake_location))
    # Y_init = np.array([func_model(start_location)[0], func_model(fake_location)[0]])

    Y_init = np.vstack((f([start_location]), f([fake_location])))
    # print Y_init
    X_step = X_init
    Y_step = Y_init

    max_iter = 20 / batch_size  # evaluation budget
    bo_data_size = 0
    for i in range(max_iter):
        current_lookahead = 20 - i * batch_size
        current_lookahead = 20
        print "lookahead %d" % current_lookahead
        current_data_size = 2 + batch_size * (i + 1)
        while bo_data_size < current_data_size:
            myBopt = GPyOpt.methods.BayesianOptimization(f=f,  # function to optimize
                                                         domain=domain,
                                                         X=X_step,
                                                         Y=Y_step,
                                                         initial_design_numdata=-1,
                                                         acquisition_type='MPI',
                                                         # exact_feval=True,
                                                         normalize_Y=None,
                                                         optimize_restarts=10,
                                                         # acquisition_weight=2,
                                                         evaluator_type='local_penalization',
                                                         batch_size=current_lookahead,
                                                         num_cores=4,
                                                         de_duplication=True)
            # maximize=True)
            myBopt.run_optimization(1)
            myBopt._print_convergence()
            bo_data_size = myBopt.X.shape[0]
            print "BO data size %d" % bo_data_size

        # print "current obtained data size %d" % current_data_size
        X_step = myBopt.X[:current_data_size, :]
        Y_step = np.array([f(X_step[k:k+1, :])[0] for k in range(current_data_size)])
        # print Y_step
        # print myBopt.Y
    return myBopt.X


def BOLoop(start_location, fake_location, domain, f, batch_size):
    X_init = np.vstack((start_location, fake_location))
    # Y_init = np.array([func_model(start_location)[0], func_model(fake_location)[0]])
    Y_init = np.vstack((f([start_location]), f([fake_location])))
    # print Y_init
    myBopt = GPyOpt.methods.BayesianOptimization(f=f,  # function to optimize
                                                 domain=domain,
                                                 X=X_init,
                                                 Y=Y_init,
                                                 initial_design_numdata=5,
                                                 acquisition_type='MPI',
                                                 # exact_feval=True,
                                                 normalize_Y=True,
                                                 optimize_restarts=10,
                                                 # acquisition_weight=2,
                                                 evaluator_type='local_penalization',
                                                 batch_size=batch_size,
                                                 num_cores=4,
                                                 de_duplication=True)
    max_iter = 20 / batch_size  # evaluation budget
    myBopt.run_optimization(max_iter)
    myBopt._print_convergence()
    print myBopt.X.shape
    # print myBopt.x_opt, func_model([myBopt.x_opt])
    return myBopt.X


def PerformBOForOneSeed(seed, m, my_save_folder_root, batch_size):
    def func_model(location):
        location = location[0]
        return  - np.array([[m(location)]])

    save_folder = my_save_folder_root + "seed" + str(seed) + "/"
    start_location = m.LoadRandomLocation(save_folder)
    informative_locations = m.locations[m.informative_locations_indexes, :]
    domain = [{'name': 'locations', 'type': 'bandit', 'domain': informative_locations}]
    X_ans = np.zeros((2, 0))

    while X_ans.shape[0] != 22:
        # doesn't work with only one starting location
        neighb = m.GetNeighbours(start_location)
        fake_location = choice(neighb)

        while np.array_equal(start_location, fake_location):
            fake_location = choice(neighb)

        print start_location, fake_location
        # fake_location = choice(m.locations[m.informative_locations_indexes, :])
        # fake_location = np.array([49.0, 48.0])
        # print fake_location,start_location
        # fake_location = m.GetRandomStartLocation(batch_size=batch_size)

        X_ans = BOLoop(start_location=start_location, fake_location=fake_location,
                        domain=domain, f=func_model, batch_size=batch_size)

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
    method_folder = save_folder + 'bbo-llp9/'
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

    time_slot = 18

    t, batch_size, num_samples = (4, 5, 300)

    filename = '../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = '../releaseTests/updated_release/road/b5-18-log/'

    model = GenerateRoadModelFromFile(filename=filename)
    for seed_0 in range(35):
        print seed_0
        PerformBOForOneSeed(seed=seed_0, m=model, my_save_folder_root=my_save_folder_root, batch_size=batch_size)
