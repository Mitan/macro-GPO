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
    m.AddTwoSidedRoads()
    m.LogTransformValues()
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


if __name__ == "__main__":
    # cannot use - cylcic linking
    file_name = './taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)
    for i in m.locations:
        print i, m.GetNeighbours(i)