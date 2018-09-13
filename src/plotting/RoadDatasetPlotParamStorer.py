import numpy as np
from src.enum.MetricsEnum import MetricsEnum


class RoadDatasetPlotParamStorer:

    def __init__(self, metric_type):

        if metric_type == MetricsEnum.TotalReward:
            self.y_label_caption = "Total normalized output measurements observed by AV"
            self.y_ticks_range = range(-1, 7)
            self.y_lim_range = [-1.5, 6]
            self.legend_loc = 2

        elif metric_type == MetricsEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.5, 4, 0.5)
            self.y_lim_range = None
            self.legend_loc = 1

            raise Exception("Wrong plotting type")
