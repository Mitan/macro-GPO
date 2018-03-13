import numpy as np

from src.enum.DatasetEnum import DatasetEnum


def get_hyper_storer(dataset_type, time_slot):
    if dataset_type == DatasetEnum.Robot:
        if time_slot == 2:
            return RobotHypersStorer_2()
        elif time_slot == 16:
            return RobotHypersStorer_16()
        else:
            raise Exception("wrong robot time slot")
    elif dataset_type == DatasetEnum.Road:
        if time_slot == 44:
            return RoadHypersStorer_Log44()
        elif time_slot == 18:
            return RoadHypersStorer_Log18()
        else:
            raise Exception("wrong taxi time slot")
    else:
        raise ValueError("Unknown dataset")


class AbstarctHypersStorer:
    def __init__(self):
        pass

    def PrintParams(self):
        print self.length_scale, self.signal_variance, self.noise_variance, self.mean_function

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
        self.length_scale = (0.25, 0.25)
        self.signal_variance = 1.0
        self.noise_variance = 0.00001
        self.mean_function = 0.0

        self.grid_gap = 0.05

        # number of samples in each dimension
        self.num_samples_grid = (50, 50)

        # upper values are not included
        self.grid_domain = ((-0.25, 2.25), (-0.25, 2.25))

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([[1.0, 1.0]])


class RoadHypersStorer_Log18(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5249, 0.5687)
        signal_cov = 0.7486
        self.signal_variance = signal_cov
        noise_cov = 0.0111
        self.noise_variance = noise_cov
        self.mean_function = 1.5673

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RoadHypersStorer_Log44(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.6276, 0.6490)
        signal_cov = 0.7969
        self.signal_variance = signal_cov ** 2
        noise_cov = 0.0117
        self.noise_variance = noise_cov ** 2
        self.mean_function = 1.4646

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RobotHypersStorer_2(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (5.139014, 9.975326)

        self.signal_variance = 0.464407

        self.noise_variance = 0.022834
        self.mean_function = 22.924200

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RobotHypersStorer_16(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (4.005779, 11.381141)

        self.signal_variance = 0.596355

        self.noise_variance = 0.059732
        self.mean_function = 17.851283

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


# unused
class RoadHypersStorer_44(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5767, 0.5941)
        self.signal_variance = 8.1167
        self.noise_variance = 0.0100
        self.mean_function = 4.6038
        """
        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))
        """
        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


# unused
class RoadHypersStorer_18(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5205, 0.5495)
        self.signal_variance = 6.0016
        self.noise_variance = 0.0100
        self.mean_function = 4.9934
        """
        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))
        """
        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])