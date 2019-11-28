import math

from src.Utils import get_rewards_regrets_latex
from src.enum.DatasetEnum import DatasetEnum
from src.enum.MetricsEnum import MetricsEnum
from src.enum.PlottingEnum import PlottingEnum
from src.metric.ResultCalculator import ResultCalculator
from src.plotting.ResultsPlotter import ResultGraphPlotter


def CalculateMetrics(metric_type,
                     plotting_type,
                     filename,
                     plot_bars,
                     seeds,
                     root_path,
                     param_storer_string):
    total_budget = 20
    h = 3
    batch_size = 4
    # iteration_list = [50, 300, 1000]
    samples = [5, 10, 20, 30, 50, 70, 100, 150]

    methods = ['h3_b{}_s{}'.format(batch_size, s) for s in samples]

    method_names = [r'$\epsilon$-M-GPO  $H = 3$ ${}$ samples.'.format(s) for s in samples]

    cut = -2
    methods = methods[:cut]
    method_names = method_names[:cut]

    output_file = root_path + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Simulated,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Simulated,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget,
                                         param_storer_string=param_storer_string)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)


    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    return results


def GetSimulatedTotalRewards(seeds, root_path, filename, param_storer_string=None):
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename=filename,
                            plot_bars=False,
                            seeds=seeds,
                            root_path=root_path,
                            param_storer_string=param_storer_string)


def GetSimulatedTotalRegrets(seeds, root_path, filename, param_storer_string=None):
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename=filename,
                            plot_bars=False,
                            seeds=seeds,
                            root_path=root_path,
                            param_storer_string=param_storer_string)


if __name__ == "__main__":
    batch_size = 4

    root_path = '../../tests/simulated_h3_b4/'
    seeds = list(set(range(66, 101, 7)) - set([19]))
    rewards = GetSimulatedTotalRewards(root_path=root_path,
                                       seeds=seeds,
                                       filename='sim_i_total_rewards.eps',
                                       )
    print
    regrets = GetSimulatedTotalRegrets(root_path=root_path,
                                       seeds=seeds,
                                       filename='sim_i_simple_regrets.eps',
                                       )