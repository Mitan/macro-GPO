import numpy as np
from src.enum.PlottingEnum import PlottingEnum


class RoadDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Avg. normalized output measurements observed by AV"
            self.y_ticks_range = np.arange(-0.3, 0.3, 0.1)
            self.y_lim_range = [0.03, 0.35]
            self.legend_loc = 4
            self.legend_size = 15

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.5, 4, 0.4)
            self.y_lim_range = [1.45, 3.2]
            self.legend_loc = 1
            self.legend_size = 16

        elif plotting_type == PlottingEnum.SimpleRegretFull:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.5, 4, 0.4)
            self.y_lim_range = [1.45, 3.7]
            self.legend_loc = 1
            self.legend_size = 18

        elif plotting_type == PlottingEnum.AverageRewardBeta:
            self.y_label_caption = "Avg. normalized output measurements observed by AV"
            self.y_ticks_range = np.arange(-0.3, 0.3, 0.1)
            self.y_lim_range = [-0.31, 0.27]
            self.legend_loc = 4
            self.legend_size = 20

        elif plotting_type == PlottingEnum.AverageRewardFull:
            self.y_label_caption = "Avg. normalized output measurements observed by AV"
            self.y_ticks_range = np.arange(-0.3, 0.3, 0.1)
            self.y_lim_range = [-0.31, 0.28]
            self.legend_loc = 4
            self.legend_size = 17

        else:
            raise Exception("Wrong plotting type %s" % plotting_type)
