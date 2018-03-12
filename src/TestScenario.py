import os

from DatasetUtils import GenerateRoadModelFromFile
from src.enum.MethodEnum import Methods
from TreePlanTester import testWithFixedParameters

# from ResultsPlotter import PlotData
from src.model.DatasetGenerator import DatasetGenerator
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def TestScenario_MLE(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot, filename):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    m = GenerateRoadModelFromFile(filename)
    m.LoadSelectedMacroactions(save_folder, batch_size)

    start_location = m.LoadRandomLocation(save_folder)

    h = 4

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    mle = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.MLE, horizon=h,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h" + str(h) + "/",
                                  num_samples=num_samples, batch_size=batch_size,
                                  start_location=start_location)

    method_name = 'MLE H = ' + str(h)

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')
    output_rewards.close()


def TestScenario_LP(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Robot, dataset_mode=DatasetModeEnum.Load,
                                         time_slot=time_slot)
    m = dataset_generator.get_dataset_model()
    m.LoadSelectedMacroactions(save_folder, batch_size)
    # m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)

    lp = testWithFixedParameters(model=m, method=Methods.LP, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "my_lp/",
                                 num_samples=num_samples,
                                 batch_size=batch_size,
                                 start_location=start_location)


def TestScenario_PE_qEI_BUCB(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    """
    m = GenerateRoadModelFromFile(filename)
    m.LoadSelectedMacroactions(save_folder, batch_size)

    start_location = m.LoadRandomLocation(save_folder)
    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)
    """
    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Robot, dataset_mode=DatasetModeEnum.Load,
                                         time_slot=time_slot)
    m = dataset_generator.get_dataset_model()
    m.LoadSelectedMacroactions(save_folder, batch_size)
    # m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)
    """
    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)
    """
    qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "qEI/",
                                  num_samples=num_samples, batch_size=batch_size,
                                  start_location=start_location)
    method_name = 'r_QEI'
    # output_rewards.write(method_name + '\n')
    # output_rewards.write(str(qEI) + '\n')
    """

    PE = testWithFixedParameters(model=m, method=Methods.BucbPE, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "fixed_pe/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'FIXED-BUCB-PE'

    # output_rewards.write(method_name + '\n')
    # output_rewards.write(str(PE) + '\n')

    bucb = testWithFixedParameters(model=m, method=Methods.BUCB, horizon=1,
                                   num_timesteps_test=time_steps,
                                   save_folder=save_folder + "gp-bucb/",
                                   num_samples=num_samples, batch_size=batch_size,
                                   start_location=start_location)

    method_name = 'BUCB'
    """
    # output_rewards.write(method_name + '\n')
    # output_rewards.write(str(bucb) + '\n')

    # output_rewards.close()


def TestScenario_EI_PI(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    """
    m = GenerateRoadModelFromFile(filename)
    m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)
    """

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Robot, dataset_mode=DatasetModeEnum.Load,
                                         time_slot=time_slot)
    m = dataset_generator.get_dataset_model()

    # m.LoadSelectedMacroactions(save_folder, batch_size)
    m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)
    h = -1

    testWithFixedParameters(model=m, method=Methods.PI, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "pi/",
                            num_samples=num_samples, batch_size=batch_size,
                            start_location=start_location)

    testWithFixedParameters(model=m, method=Methods.EI, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "ei/",
                            num_samples=num_samples, batch_size=batch_size,
                            start_location=start_location)


def TestScenario_2Full(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot, coords_filename,
                       data_filename, neighbours_filename):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)
    # m.LoadSelectedMacroactions(save_folder, batch_size)
    m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)

    h = 2

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    h2 = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.Anytime, horizon=h,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "anytime_h" + str(h) + "_full/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'Anytime Full 21 H = ' + str(h)

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h2) + '\n')
    output_rewards.close()


def TestScenario_justH4(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot, data_filename,
                        coords_filename, neighbours_filename, h):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # m = GenerateRoadModelFromFile(filename)
    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)
    m.LoadSelectedMacroactions(save_folder, batch_size)
    # m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    h4 = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.Anytime, horizon=4,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "n_anytime_h" + str(4) + "_" + str(num_samples) + "/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'Anytime H = ' + str(h) + ' ' + str(num_samples)

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h4) + '\n')
    output_rewards.close()


def TestScenario_H4(my_save_folder_root, seed, time_steps, num_samples, batch_size, time_slot, data_filename,
                    coords_filename, neighbours_filename, h):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # m = GenerateRoadModelFromFile(filename)
    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)
    m.LoadSelectedMacroactions(save_folder, batch_size)
    # m.SelectMacroActions(folder_name=save_folder, batch_size=batch_size, select_all=True)

    start_location = m.LoadRandomLocation(save_folder)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    # h = 4

    # h_start = 1
    # h_end = 3
    # for h in range(h_start, h_end + 1):
    h_end = 3
    """
    for current_h in range(h_end, h_end +1):
        # print h
        method_name = 'Anytime H = ' + str(current_h)
        current_h_result = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.Anytime, horizon=current_h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "anytime_h" + str(current_h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size,
                                                   start_location=start_location)

        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')
    """

    PE = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.BucbPE, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "pe/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'FIXED-BUCB-PE'

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(PE) + '\n')

    bucb = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.BUCB, horizon=1,
                                   num_timesteps_test=time_steps,
                                   save_folder=save_folder + "gp-bucb/",
                                   num_samples=num_samples, batch_size=batch_size,
                                   start_location=start_location)

    method_name = 'BUCB'

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(bucb) + '\n')

    method_name = 'MLE H = 4'
    mle = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.MLE, horizon=4,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h4/",
                                  num_samples=num_samples, batch_size=batch_size,
                                  start_location=start_location)
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')

    h4 = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.Anytime, horizon=4,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "new_anytime_h" + str(4) + "_" + str(num_samples) + "/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'Anytime H = ' + str(h) + ' ' + str(num_samples)

    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h4) + '\n')
    output_rewards.close()


def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size, time_slot, data_filename=None,
                 coords_filename=None, neighbours_filename=None):
    result_graphs = []

    # eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    # file for storing reward histories
    # so that later we can plot only some of them

    output_rewards = open(filename_rewards, append_write)

    # Generation of simulated model
    """
    if filename is not None:
        m = GenerateModelFromFile(filename)

    else:
        hyper_storer = SimulatedHyperStorer()
        m = GenerateSimulatedModel(length_scale=np.array(hyper_storer.length_scale),
                                   signal_variance=hyper_storer.signal_variance,
                                   seed=seed, noise_variance=hyper_storer.noise_variance, save_folder=save_folder,
                                   predict_range=hyper_storer.grid_domain, num_samples=hyper_storer.num_samples_grid,
                                   mean_function=hyper_storer.mean_function)
    """

    # m = GenerateRoadModelFromFile(filename)
    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)

    # todo note
    m.SelectMacroActions(batch_size, save_folder)

    start_location = m.GetRandomStartLocation(batch_size=batch_size)

    with  open(save_folder + "start_location.txt", 'w') as f:
        f.write(str(start_location[0]) + " " + str(start_location[1]))

    # simulated
    """
    if batch_size > 1:
        method_name = 'qEI'
        qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "qEI/",
                                      num_samples=num_samples, batch_size=batch_size)
        result_graphs.append([method_name, qEI])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(qEI) + '\n')

    method_name = 'Myopic DB-GP-UCB'
    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=1, num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, myopic_ucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(myopic_ucb) + '\n')

    # for h in range(2, h_max+1):
    for h in range(h_max, 1, -1):
        # print h
        method_name = 'H = ' + str(h)
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')



    method_name = 'MLE H = 4'
    mle = testWithFixedParameters(model=m, method=Methods.MLE, horizon=4, num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h4/",
                                  num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, mle])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')



    method_name = 'Anytime H = 3'
    anytime = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "anytime_h3/",
                                      num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, anytime])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(anytime) + '\n')

    method_name = 'BUCB-PE'
    bucb = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.BucbPE, horizon=1,
                                   num_timesteps_test=time_steps,
                                   save_folder=save_folder + "bucb-pe/",
                                   num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, bucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(bucb) + '\n')
    """
    # real-world

    if batch_size > 1:
        method_name = 'qEI'
        qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "qEI/",
                                      num_samples=num_samples, batch_size=batch_size, start_location=start_location,
                                      time_slot=time_slot)
        result_graphs.append([method_name, qEI])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(qEI) + '\n')

    method_name = 'Myopic DB-GP-UCB'
    myopic_ucb = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.MyopicUCB, horizon=1,
                                         num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, myopic_ucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(myopic_ucb) + '\n')

    PE = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.BucbPE, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "new_pe/",
                                 num_samples=num_samples, batch_size=batch_size,
                                 start_location=start_location)

    method_name = 'NEW-BUCB-PE'
    result_graphs.append([method_name, PE])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(PE) + '\n')

    method_name = 'MLE H = 4'
    mle = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.MLE, horizon=4,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h4/",
                                  num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, mle])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')

    # for h in range(2, h_max+1):
    # for h in range(2, h_max + 1):
    for h in range(h_max, 1, -1):
        # print h
        method_name = 'Anytime H = ' + str(h)
        current_h_result = testWithFixedParameters(time_slot=time_slot, model=m, method=Methods.Anytime, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "anytime_h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size,
                                                   start_location=start_location)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')

    output_rewards.close()
    # PlotData(results=result_graphs, output_file_name=save_folder+ 'results.eps', isRoad=False, isTotalReward=True)


def TestScenario_Beta(my_save_folder_root, seed, time_steps, num_samples, batch_size, beta, test_horizon,
                      time_slot,
                      coords_filename,
                      data_filename, neighbours_filename):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)

    m.LoadSelectedMacroactions(save_folder, batch_size)

    start_location = m.LoadRandomLocation(save_folder)

    # for beta in beta_list:
    method_name = 'beta = ' + str(beta)
    current_h_result = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=test_horizon,
                                               num_timesteps_test=time_steps,
                                               save_folder=save_folder + "beta" + str(beta) + "/",
                                               num_samples=num_samples, batch_size=batch_size, beta=beta,
                                               start_location=start_location, time_slot=time_slot)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(current_h_result) + '\n')

    output_rewards.close()
    # PlotData(result_graphs, save_folder)
