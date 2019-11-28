import numpy as np

from src.enum.CovarianceEnum import CovarianceEnum
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
    elif dataset_type == DatasetEnum.Simulated:
        return SimulatedHyperStorer()
    if dataset_type == DatasetEnum.Branin:
        return BraininHyperStorer()
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

        # self.PrintParams()
    """
    def GetInitialPhysicalState(self, start_location):
        return np.array([[1.0, 1.0]])
    """

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
    """
    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])
    """

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
    """
    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])
    """

class RobotHypersStorer_2(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (5.139014, 9.975326)

        self.signal_variance = 0.464407

        self.noise_variance = 0.022834
        self.mean_function = 22.924200

        self.PrintParams()
    """
    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])
    """

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
    """
    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])
    """

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


class BraininHyperStorer(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.type = CovarianceEnum.SquareExponential
        """
        self.length_scale = (4.2551 * 4.32877364, 18.898 * 4.32877364)
        self.signal_variance = 93024.441
        self.noise_variance = 1.0
        self.length_scale = (5.0, 5.0)
        self.signal_variance = 3.0
        self.noise_variance = 1.0
        self.mean_function =0.0

        self.empirical_mean = 0.001
        self.max_value = 1.04001929678
        
        # 400 with noise
        self.length_scale = (4.55891903e+00, 2.41301070e+01)
        self.signal_variance = 54.1469461108397
        self.noise_variance = 0.01
        self.mean_function = 0.0

        self.empirical_mean = 0.000187585472341
        self.max_value = 1.03275606571
        """
        # 1600 ok
        # branin 1600
        self.length_scale = (4.85609470e+00, 4.46323187e+01)
        self.signal_variance = 428.88925324174573
        self.noise_variance = 0.001
        self.mean_function = 0.0

        self.empirical_mean = 1.61137748492e-05
        self.max_value = 1.0417132636

        # camel
        self.length_scale = (0.84379577, 0.74023375)
        self.signal_variance = 1.18474464
        self.noise_variance =  0.01072684

        self.mean_function = 1.30266168223e-16
        self.empirical_mean = -0.00549505294253

        self.max_value = 2.08485259411

        # # goldstein
        # self.length_scale = (0.48103349, 0.37723533)
        # self.signal_variance = 0.62406836
        # self.noise_variance =  0.01720424
        #
        # self.mean_function = -3.73034936274e-16
        # self.empirical_mean = -0.00285519952058
        #
        # self.max_value = 2.55791162941
        #
        # # boha
        # self.length_scale = (8.27664496e+01, 4.49769834e+01)
        # self.signal_variance = 1.54177002e+00
        # self.noise_variance = 1.04647905e-02
        #
        # self.mean_function = -7.1054273576e-17
        # self.empirical_mean = -0.00285519952058
        #
        # self.max_value = 2.64778379782
