from StringIO import StringIO
import numpy as np

from GaussianProcess import SquareExponential, GaussianProcess, MapValueDict
from RoadMapValueDict import RoadMapValueDict


def GenerateSimulatedModel(length_scale, signal_variance, noise_variance, save_folder, seed, predict_range,
                           num_samples, mean_function):
    covariance_function = SquareExponential(length_scale, signal_variance=signal_variance,
                                            noise_variance=noise_variance)
    # Generate a drawn vector from GP with noise
    gpgen = GaussianProcess(covariance_function, mean_function=mean_function)
    m = gpgen.GPGenerate(predict_range=predict_range, num_samples=num_samples, seed=seed, noiseVariance=noise_variance)
    # write the dataset to file
    m.WriteToFile(save_folder + "dataset.txt")
    return m


def GenerateModelFromFile(filename):
    data = np.genfromtxt(filename)
    locs = data[:, :-1]
    vals = data[:, -1]
    return MapValueDict(locs, vals)


def GenerateRoadModelFromFile(filename):
    m = RoadMapValueDict(filename)
    # m.AddTwoSidedRoads()
    # m.LogTransformValues()
    return m


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

# todo refact this and next method in one
def GetMaxValues(measurements, batch_size):
    assert len(measurements) == 21
    number_of_steps = 20 / batch_size
    max_values = []
    # initial value before planning
    max_values.append(measurements[0])
    for i in range(number_of_steps):
        # first batch_size * i + 1 (initial) point
        # since i starts from zero, need to take i+1
        after_i_step_points = batch_size * (i + 1) + 1
        current_max = max(measurements[:after_i_step_points])
        max_values.append(current_max)
    return np.array(max_values)


def GetAccumulatedRewards(measurements, batch_size):
    assert len(measurements) == 21
    number_of_steps = 20 / batch_size
    acc_rewards = []
    # initial value before planning
    acc_rewards.append(measurements[0])
    for i in range(number_of_steps):
        # first batch_size * i + 1 (initial) point
        # since i starts from zero, need to take i+1
        after_i_step_points = batch_size * (i + 1) + 1
        current_reward = sum(measurements[:after_i_step_points])
        acc_rewards.append(current_reward)
    return np.array(acc_rewards)


# get all the measurements collected by the method including initial value
# these values are not normalized
def GetAllMeasurements(root_folder, method_name, batch_size):
    n_steps = 20 / batch_size
    i = n_steps - 1

    step_file_name = root_folder + method_name + '/step' + str(i) + '.txt'
    lines = open(step_file_name).readlines()
    first_line_index = 1 + batch_size + 1 + (1 + batch_size * (i + 1)) + 1
    last_line_index = -1
    stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index])
    joined_lines = " ".join(stripped_lines)
    assert joined_lines[0] == '['
    assert joined_lines[-1] == ']'
    a = StringIO(joined_lines[1:-1])

    # all measurements obtained by the robot till that step
    measurements = np.genfromtxt(a)

    assert measurements.shape[0] == 21

    # assert we parsed them all as numbers
    assert not np.isnan(measurements).any()

    return measurements.tolist()


if __name__ == "__main__":
    # cannot use - cylcic linking
    file_name = './taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)
    for i in m.locations:
        print i, m.GetNeighbours(i)