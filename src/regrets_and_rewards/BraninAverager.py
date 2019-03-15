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
                     plot_bars):
    batch_size = 1
    total_budget = 20

    # seeds = list(set(range(0, 55)) - set([6, 11, 41, 43]))
    """
    seeds = list(set(range(0, 55)) - set([ 50,  48, 40, 33, 32]))
    root_path = '../../tests/branin/branin400/'
    methods = ['h4_b1_20','h2_b1_20', 'ei']

    seeds = list(set(range(0, 86)) - set([63, 55, 44, 31, 7, 3]))
    root_path = '../../tests/branin/branin_30_400/'
    methods = ['h4_b1_30','h2_b1_30', 'ei']
    methods = ['h4_b1_30', 'ei']
    
    seeds = list(set(range(0, 80, 2)) - set([]))
    root_path = '../../tests/branin/branin_100_400/'
    methods = ['h3_b1_100','h2_b1_100', 'ei']
    """
    seeds = list(set(range(0, 62)) - set([]))
    root_path = '../../tests/branin/branin_new_1600/'
    methods = ['h4_b1_20', 'ei']
    print(len(seeds))
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'EI']
    # method_names = [
    #                 r'$\epsilon$-Macro-GPO  $H = 4$',
    #                 'EI']
    output_file = '../../result_graphs/eps/branin/' + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Branin,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)
    """
    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Branin,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)
    """
    s = math.sqrt(428.88925324174573)
    print (21.0982 - 18.6664) / s

    print (3.9028 - 3.3118) / s

    for result in results:
        pass
        # print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    return results





def GetSimulatedTotalRewards():
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename='branin_total_rewards.eps',
                            plot_bars=False)


def GetSimulatedTotalRegrets():
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename='branin_simple_regrets.eps',
                            plot_bars=False)


if __name__ == "__main__":
    rewards = GetSimulatedTotalRewards()
    regrets = GetSimulatedTotalRegrets()
