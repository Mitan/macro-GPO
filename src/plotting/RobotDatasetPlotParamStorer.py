from src.enum.PlottingEnum import PlottingEnum
import numpy as np


class RobotDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Avg. normalized output measurements observed by mobile robot"
            self.y_ticks_range = np.arange(0.0, 0.65, 0.1)
            self.y_lim_range = [0.075, 0.65]
            self.legend_loc = 2
            self.legend_size = 13

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(0.4, 1.6, 0.2)
            self.y_lim_range = [0.35, 1.5]
            self.legend_loc = 1
            self.legend_size = 14.5

        elif plotting_type == PlottingEnum.AverageRewardFull:
                self.y_label_caption = "Avg. normalized output measurements observed by mobile robot"
                self.y_ticks_range = np.arange(0.0, 0.65, 0.1)
                self.y_lim_range = [0.17, 0.65]
                self.legend_loc = 2
                self.legend_size = 16

        elif plotting_type == PlottingEnum.SimpleRegretFull:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(0.4, 1.6, 0.2)
            self.y_lim_range = [0.35, 1.5]
            self.legend_loc = 1
            self.legend_size = 17

        elif plotting_type == PlottingEnum.AverageRewardBeta:
            self.y_label_caption = "Avg. normalized output measurements observed by mobile robot"
            self.y_ticks_range = np.arange(0.0, 0.65, 0.1)
            self.y_lim_range = [0.09, 0.65]
            self.legend_loc = 2
            self.legend_size = 22
        else:
            raise Exception("Wrong plotting type")
