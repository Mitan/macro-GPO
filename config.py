from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.enum.MethodEnum import Methods
from src.enum.MetricsEnum import MetricsEnum
from src.run.MethodDescriptor import MethodDescriptor


class Config:

    def __init__(self):
        pass

    # batch size
    BATCH_SIZE = 2

    # type of dataset - currently Simulated, Road and Robot are supported
    DATASET_TYPE = DatasetEnum.Simulated

    # load or generate dataset. If "Load" is selected, the root dataset folder should be specified
    DATASET_MODE = DatasetModeEnum.Load

    # the folder containing the dataset
    # if None, the current seed folder will be used (e.g. for loading the simulated dataset)
    DATASET_ROOT_FOLDER = None

    RESULTS_SAVE_ROOT_FOLDER = './tests/'
    RESULTS_SAVE_ROOT_FOLDER = './1/simulated/'

    # total budget of function evaluations
    TOTAL_BUDGET = 20

    # seeds for evaluation
    SEEDS = range(3)

    # threshold for selecting the number of random macro-actions. Selected when
    MA_THRESHOLD = 20

    # list of methods to run
    METHODS = [MethodDescriptor(method_type=Methods.Exact,
                                h=3,
                                beta=0.0,
                                num_samples=40),
               MethodDescriptor(method_type=Methods.Exact,
                                h=2,
                                beta=0.0,
                                num_samples=40),
               MethodDescriptor(method_type=Methods.Exact,
                                h=1,
                                beta=0.0,
                                num_samples=40)
               ]
    # metrics to calculate
    METRICS_LIST = (MetricsEnum.AverageTotalReward, MetricsEnum.SimpleRegret)

    #  plot error bars in graphs or not
    PLOT_BARS = False
