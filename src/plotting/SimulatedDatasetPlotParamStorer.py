import numpy as np
from src.enum.MetricsEnum import MetricsEnum


class SimulatedDatasetPlotParamStorer:

    def __init__(self, metric_type):

        if metric_type == MetricsEnum.AverageTotalReward:
            self.y_label_caption = "Average normalized output measurements observed by AUV"
            self.y_ticks_range = np.arange(0, 0.75, 0.05)
            self.y_lim_range = [-0.07, 0.7]
            self.legend_loc = 2

        elif metric_type == MetricsEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.2, 2.8, 0.2)
            self.y_lim_range = [1.2, 2.65]
            self.legend_loc = 1
        else:
            raise Exception("Wrong plotting type")
