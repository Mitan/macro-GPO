from DataSetFunctionDefinitions import __Ackley, __Cosines, _Branin, _Eggholder, _Griewank, _HolderTable, _McCormick, \
    _SixCamel, __DropWave, \
    __Shubert, LogKDataset, LogPDataset


# class with hardcoded parameters of simulated functions
class __DatasetInfo:
    def __init__(self, f, mean, lengthscale, signal_variance, noise_variance, domain, grid_gap, bad_places=None):
        self.lengthscale = lengthscale
        self.grid_gap = grid_gap
        self.domain = domain
        self.simulated_function = f
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.mean = mean
        self.name = f.__name__
        self.bad_places = bad_places


def AckleyInfo():
    return __DatasetInfo(f=__Ackley, lengthscale=(4.97859606259, 4.97858531881),
                         signal_variance=0.151286346398, noise_variance=0.00621336727951, mean=2.43787748468,
                         domain=((-5.0, 5.0), (-5.0, 5.0)), grid_gap=0.5)


def CosinesInfo():
    return __DatasetInfo(f=__Cosines, lengthscale=(0.12605123651, 0.126051232038),
                         signal_variance=0.0198660061591, noise_variance=0.0001, mean=0.940527042428,
                         domain=((0.0, 1.0), (0.0, 1.0)), grid_gap=0.05)


def DropWaveInfo():
    return __DatasetInfo(f=__DropWave, lengthscale=(0.33928922958, 0.339289285926),
                         signal_variance=26.199383292, noise_variance=0.001, mean=0.433088851736,

                         domain=((-1.0, 1.0), (-1.0, 1.0)), grid_gap=0.10)


"""
def ShubertInfo():
    return __SimulatedFunctionInfo(f=__Shubert, lengthscale=( 0.589966057898, 0.592535157371),
                                   signal_variance=2521081.3233, noise_variance=0.01, mean=0.0260921231097,
                                   domain=((-2.0, 2.0), (-2.0, 2.0)), grid_gap=0.2)
"""


def BraninInfo():
    return __DatasetInfo(f=_Branin, lengthscale=(0.817672119372, 2.20113627131),
                         signal_variance=0.26708090617, noise_variance=0.001, mean=3.70878384532,
                         domain=((-5.0, 10.0), (0.0, 15.0)), grid_gap=0.75)


def GriewankInfo():
    return __DatasetInfo(f=_Griewank, lengthscale=(2.80253742229, 3.79926020589),
                         signal_variance=26.4682341649, noise_variance=0.1, mean=0.984014997574,
                         domain=((-5, 5), (-5, 5)), grid_gap=0.5)


def McCormickInfo():
    return __DatasetInfo(f=_McCormick, lengthscale=(0.710953395473, 1.11973978551),
                         signal_variance=0.441333703172, noise_variance=0.001, mean=1.98094886523,
                         domain=((-1.0, 4.0), (-1.0, 4.0)), grid_gap=0.25)


def SixCamelInfo():
    return __DatasetInfo(f=_SixCamel, lengthscale=(0.397748187414, 0.300164660582),
                         signal_variance=0.592278564153, noise_variance=6.50838976475e-05, mean=1.6232484575,
                         domain=((-2.0, 2.0), (-2.0, 2.0)), grid_gap=0.2)


def HolderTableInfo():
    return __DatasetInfo(f=_HolderTable, lengthscale=(1.04938884147, 1.16990619532),
                         signal_variance=1.78932992583, noise_variance=0.000108666419245, mean=0.537306851025,
                         domain=((-10.0, 10.0), (-10.0, 10.0)), grid_gap=1.0)


###### Real world

def Log_K_Info():
    dataset = LogKDataset()
    f = lambda t: dataset(t)
    """
    set = __DatasetInfo(f=f, lengthscale=(1.246, 2.377),
                         signal_variance=0.0595, noise_variance=0.0142, mean=3.2836492219,
                         domain=((5.0, 19.0), (6.0, 18.0)), grid_gap=1.0,
                         bad_places=[[5.0, 6.0, ], [5., 7.], [5., 8.], [6.,6.], [7., 6.], [8., 6.],[18., 17.]])
    """
    set = __DatasetInfo(f=f, lengthscale=(1.246, 2.378),
                         signal_variance=0.0389, noise_variance=0.0159, mean=3.2836492219,
                         domain=((5.0, 19.0), (6.0, 18.0)), grid_gap=1.0,
                         bad_places=[[5.0, 6.0, ], [5., 7.], [5., 8.], [6.,6.], [7., 6.], [8., 6.],[18., 17.]])
    set.name = "LogK"
    return set

#(1.246389172612413, 2.3779138261618842, 0.015887914883705639, 0.0389499196745028)

def Log_P_Info():
    pass
    """
    f = LogPDataset
    return __DatasetInfo(f=f, lengthscale=(1.246, 2.377),
                         signal_variance=0.0595, noise_variance=0.0142, mean=3.2836492219,
                         domain=((5.0, 19.0), (7.0, 19.0)), grid_gap=1.0,
                         bad_places=[[5.0, 6.0, ], [5., 7.], [5., 8.], [6.,6.], [7., 6.], [8., 6.],[18., 17.]])
    """