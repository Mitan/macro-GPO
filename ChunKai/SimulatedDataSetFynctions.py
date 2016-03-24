from SimulatedDataSetsHypers import __Ackley, __Cosines


# class with hardcoded parameters of simulated functions
class __SimulatedFunctionInfo:
    def __init__(self, f, mean, lengthscale, signal_variance, noise_variance, domain, grid_gap):
        self.lengthscale = lengthscale
        self.grid_gap = grid_gap
        self.domain = domain
        self.simulated_function = f
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.mean = mean
        self.name = f.__name__


def AckleyInfo():
    return __SimulatedFunctionInfo(f=__Ackley, lengthscale=(4.97859606259, 4.97858531881),
                                   signal_variance=0.151286346398, noise_variance=0.00621336727951, mean=2.43787748468,
                                   domain=((-5, 5), (-5, 5)), grid_gap=0.5)


def CosinesInfo():
    return __SimulatedFunctionInfo(f=__Cosines, lengthscale=(0.12605123651, 0.126051232038),
                                   signal_variance=0.0198660061591, noise_variance=1.23195228302e-15, mean=0.940527042428,
                                   domain=((0.0, 1.0), (0.0, 1.0)), grid_gap=0.05)
