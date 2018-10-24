from StringIO import StringIO

import numpy as np

from src.enum.MetricsEnum import MetricsEnum


class MethodResultsExtractor:

    def __init__(self, batch_size, method, metric_type, total_budget):
        self.total_budget = total_budget
        self.metric_type = metric_type
        self.method = method
        self.batch_size = batch_size
        self.num_steps = self.total_budget / batch_size

    # get all the measurements collected by the method including initial value
    # these values are not normalized
    def _get_all_measurements(self, root_folder):
        i = self.num_steps - 1

        step_file_name = root_folder + self.method + '/step' + str(i) + '.txt'
        lines = open(step_file_name).readlines()
        first_line_index = 1 + self.batch_size + 1 + (1 + self.batch_size * (i + 1)) + 1

        # hack to match format:
        # later added number of nodes in counting
        if lines[-2][0] == 'T':
            last_line_index = -2
        else:
            last_line_index = - 1
        stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index])
        joined_lines = " ".join(stripped_lines)
        assert joined_lines[0] == '['
        assert joined_lines[-1] == ']'
        a = StringIO(joined_lines[1:-1])

        # all measurements obtained by the robot till that step
        measurements = np.genfromtxt(a)

        assert measurements.shape[0] == self.total_budget + 1

        # assert we parsed them all as numbers
        assert not np.isnan(measurements).any()

        return measurements.tolist()

    def get_results(self, root_folder):
        measurements = self._get_all_measurements(root_folder)
        assert len(measurements) == self.total_budget + 1
        # initial value before planning
        results = [measurements[0]]

        for i in range(self.num_steps):
            # first batch_size * i + 1 (initial) point
            # since i starts from zero, need to take i+1
            after_i_step_points = self.batch_size * (i + 1) + 1
            current_result = self.__calculate_metric(measurements[:after_i_step_points], self.metric_type)
            results.append(current_result)
        return np.array(results)

    @staticmethod
    def __calculate_metric(measurements, metric_type):
        if metric_type == MetricsEnum.AverageTotalReward:
            return sum(measurements)
        elif metric_type == MetricsEnum.SimpleRegret:
            return max(measurements)
        else:
            raise Exception("Unknown metric type")
