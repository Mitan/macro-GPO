import numpy as np
from src.enum.PlottingEnum import PlottingMethods


class SimulatedDatasetPlotParamStorer:

    def __init__(self, plotting_type):

        if plotting_type == PlottingMethods.TotalReward or plotting_type == PlottingMethods.TotalRewardBeta:
            self.y_label_caption = "Total normalized output measurements observed by AUV"
            self.y_ticks_range = range(-4, 14)
            self.y_lim_range = [-3.5, 11]

            self.legend_loc = 2

        elif plotting_type == PlottingMethods.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = np.arange(1.0, 3.2, 0.2)
            self.y_lim_range = None
            self.legend_loc = 1

        elif plotting_type == PlottingMethods.CumulativeRegret:
            self.y_label_caption = "Average cumulative regret"
            self.y_ticks_range = np.arange(1.0, 3.2, 0.2)
            self.y_lim_range = None
            self.legend_loc = 1
        else:
            raise Exception("Wrong plotting type")
