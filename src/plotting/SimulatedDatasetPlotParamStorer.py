import numpy as np
from src.enum.MetricsEnum import MetricsEnum


class SimulatedDatasetPlotParamStorer:

    def __init__(self, metric_type):

        if metric_type == MetricsEnum.TotalReward:
            self.y_label_caption = "Total normalized output measurements observed by AUV"
            self.y_ticks_range = range(-4, 14)
            self.y_lim_range = [-3.5, 11]

            self.legend_loc = 2

        elif metric_type == MetricsEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.0, 3.2, 0.2)
            self.y_lim_range = [1.2, 3.0]
            self.legend_loc = 1

        elif metric_type == MetricsEnum.CumulativeRegret:
            self.y_label_caption = "Average cumulative regret"
            self.y_ticks_range = np.arange(1.0, 3.2, 0.2)
            self.y_lim_range = None
            self.legend_loc = 1
        else:
            raise Exception("Wrong plotting type")
