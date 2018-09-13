from src.enum.MetricsEnum import MetricsEnum


class RobotDatasetPlotParamStorer:

    def __init__(self, metric_type):

        if metric_type == MetricsEnum.TotalReward:
            self.y_label_caption = "Total normalized output measurements observed by mobile robot"
            self.y_ticks_range = range(0, 15)
            self.y_lim_range = [-0.5, 14]
            self.legend_loc = 2

        elif metric_type == MetricsEnum.SimpleRegret:
            self.y_label_caption = "Simple regret"
            self.y_ticks_range = None
            self.y_lim_range = None
            self.legend_loc = 1

            raise Exception("Wrong plotting type")
