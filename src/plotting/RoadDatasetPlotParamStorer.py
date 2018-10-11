import numpy as np
from src.enum.PlottingEnum import PlottingEnum


class RoadDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Average normalized output measurements observed by AV"
            self.y_ticks_range = np.arange(-0.3, 0.3, 0.05)
            self.y_lim_range = [-0.35, 0.28]
            self.legend_loc = 4

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.5, 4, 0.2)
            self.y_lim_range = [1.45, 3.7]
            self.legend_loc = 1

        elif plotting_type == PlottingEnum.AverageRewardBeta:
            self.y_label_caption = "Average normalized output measurements observed by AV"
            self.y_ticks_range = np.arange(-0.3, 0.3, 0.05)
            self.y_lim_range = [-0.31, 0.27]
            self.legend_loc = 4

        else:
            raise Exception("Wrong plotting type %s" % plotting_type)
