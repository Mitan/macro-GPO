import numpy as np
from src.enum.PlottingEnum import PlottingEnum


class BraninDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingEnum.AverageTotalReward:
            self.y_label_caption = "Avg. normalized output measurements observed by AUV"
            self.y_ticks_range = np.arange(0.15, 0.75, 0.05)
            self.y_lim_range = [0.15, 0.75]
            self.legend_loc = 2
            self.legend_size = 15

        elif plotting_type == PlottingEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(0.0, 0.8, 0.1)
            self.y_lim_range = [0.0, 0.8]
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

