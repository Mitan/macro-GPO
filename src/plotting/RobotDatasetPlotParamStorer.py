from src.enum.PlottingEnum import PlottingEnum
import numpy as np


class RobotDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Average normalized output measurements observed by mobile robot"
            self.y_ticks_range = np.arange(0.0, 0.65, 0.05)
            self.y_lim_range = [0.075, 0.65]
            self.legend_loc = 2

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(0.4, 1.6, 0.2)
            self.y_lim_range = [0.35, 1.5]
            self.legend_loc = 1

        elif plotting_type == PlottingEnum.AverageRewardFull:
            self.y_label_caption = "Average normalized output measurements observed by mobile robot"
            self.y_ticks_range = np.arange(0.0, 0.65, 0.05)
            self.y_lim_range = [0.15, 0.65]
            self.legend_loc = 2

        elif plotting_type == PlottingEnum.AverageRewardBeta:
            self.y_label_caption = "Average normalized output measurements observed by mobile robot"
            self.y_ticks_range = np.arange(0.0, 0.65, 0.05)
            self.y_lim_range = [0.09, 0.65]
            self.legend_loc = 2
        else:
            raise Exception("Wrong plotting type")
