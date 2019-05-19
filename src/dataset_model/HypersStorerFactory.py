from src.enum.CovarianceEnum import CovarianceEnum
from src.enum.DatasetEnum import DatasetEnum


def get_hyper_storer(dataset_type):
    if dataset_type == DatasetEnum.Robot:
        return RobotHypersStorer_16()
    elif dataset_type == DatasetEnum.Road:
        return RoadHypersStorer_Log18()
    elif dataset_type == DatasetEnum.Simulated:
        return SimulatedHyperStorer()
    else:
        raise ValueError("Unknown dataset")


class AbstarctHypersStorer:
    def __init__(self):
        pass

    def PrintParams(self):
        print self.length_scale, self.signal_variance, self.noise_variance, \
            self.mean_function, self.max_value, self.empirical_mean

    def PrintParamsToFile(self, file_name):
        f = open(file_name, 'w')
        f.write("mean = " + str(self.mean_function) + '\n')
        f.write("lengthscale = " + str(self.length_scale) + '\n')
        f.write("noise = " + str(self.noise_variance) + '\n')
        f.write("signal = " + str(self.signal_variance) + '\n')
        f.close()


class SimulatedHyperStorer(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.type = CovarianceEnum.SquareExponential
        self.length_scale = (0.25, 0.25)
        self.signal_variance = 1.0
        self.noise_variance = 0.00001
        self.mean_function = 0.0

        self.max_value = None
        self.empirical_mean = None


class RoadHypersStorer_Log18(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)

        self.type = CovarianceEnum.SquareExponential

        self.length_scale = (0.5249, 0.5687)
        signal_cov = 0.7486
        self.signal_variance = signal_cov
        noise_cov = 0.0111
        self.noise_variance = noise_cov
        self.mean_function = 1.5673

        self.max_value = 4.04305
        self.empirical_mean = 0.7401931727853153

        self.PrintParams()


class RobotHypersStorer_16(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)

        self.type = CovarianceEnum.SquareExponential

        self.length_scale = (4.005779, 11.381141)

        self.signal_variance = 0.596355

        self.noise_variance = 0.059732
        self.mean_function = 17.851283

        self.max_value = 19.6688
        self.empirical_mean = 17.975224827586207

        self.PrintParams()
