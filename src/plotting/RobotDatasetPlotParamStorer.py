import numpy as np
from src.enum.PlottingEnum import PlottingMethods


class RobotDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingMethods.TotalReward or plotting_type == PlottingMethods.TotalRewardBeta:
            self.y_label_caption = "Total normalized output measurements observed by mobile robot"
            self.y_ticks_range = range(0, 15)
            self.y_lim_range = [-0.5, 14]
            self.legend_loc = 2

        elif plotting_type == PlottingMethods.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = None
            self.y_lim_range = None
            self.legend_loc = 1

            raise Exception("Wrong plotting type")
