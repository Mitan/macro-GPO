import numpy as np


class AbstarctHypersStorer:
    def __init__(self):
        pass

    def PrintParams(self):
        print self.length_scale, self.signal_variance, self.noise_variance, self.mean_function

    def PrintParamsToFile(self, file_name):
        f = open(file_name, 'w')
        f.write("mean = " +str(self.mean_function) + '\n')
        f.write("lengthscale = " +str(self.length_scale) + '\n')
        f.write("noise = " +str(self.noise_variance) + '\n')
        f.write("signal = " +str(self.signal_variance) + '\n')
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


class RoadHypersStorer_44(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5767, 0.5941)
        self.signal_variance = 8.1167
        self.noise_variance = 0.0100
        self.mean_function = 4.6038

        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RoadHypersStorer_18(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5205, 0.5495)
        self.signal_variance = 6.0016
        self.noise_variance = 0.0100
        self.mean_function = 4.9934

        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RoadHypersStorer_Log18(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.5249, 0.5687)
        self.signal_variance = 0.7486
        self.noise_variance = 0.0111
        self.mean_function = 1.5673

        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])


class RoadHypersStorer_Log44(AbstarctHypersStorer):
    def __init__(self):
        AbstarctHypersStorer.__init__(self)
        self.length_scale = (0.6276, 0.6490)
        signal_cov = 0.7969
        self.signal_variance = signal_cov**2
        noise_cov = 0.0117
        self.noise_variance = noise_cov**2
        self.mean_function = 1.4646

        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))

        self.PrintParams()

    def GetInitialPhysicalState(self, start_location):
        return np.array([start_location])
