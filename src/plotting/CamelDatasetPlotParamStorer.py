import numpy as np
from src.enum.PlottingEnum import PlottingEnum


class CamelDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Avg. normalized output measurements observed by AUV"
            self.y_ticks_range = np.arange(0.0, 1.0, 0.1)
            self.y_lim_range = [0.0, 0.9]
            self.legend_loc = 4
            self.legend_size = 12

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(0.0, 2.0, 0.2)
            self.y_lim_range = [0.5, 2.1]
            self.legend_loc = 1
            self.legend_size = 13

        # elif plotting_type == PlottingEnum.AverageRewardBeta:
        #     self.y_label_caption = "Avg. normalized output measurements observed by AUV"
        #     self.y_ticks_range = np.arange(0, 0.75, 0.1)
        #     self.y_lim_range = [-0.1, 0.67]
        #     self.legend_loc = 2
        #     self.legend_size = 20

        else:
            raise Exception("Wrong plotting type")

