import numpy as np
from src.enum.PlottingEnum import PlottingEnum


class SimulatedDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Average normalized output measurements observed by AUV"
            self.y_ticks_range = np.arange(0, 0.75, 0.1)
            self.y_lim_range = [-0.1, 0.67]
            self.legend_loc = 2

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.2, 2.8, 0.2)
            self.y_lim_range = [1.15, 2.65]
            self.legend_loc = 1
        else:
            raise Exception("Wrong plotting type")
