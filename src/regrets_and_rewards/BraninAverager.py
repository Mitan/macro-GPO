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

    methods = ['h4_b{}_s{}'.format(batch_size, num_samples),
               'h3_b{}_s{}'.format(batch_size, num_samples),
               'h2_b{}_s{}'.format(batch_size, num_samples),
               'h1_b{}_s{}'.format(batch_size, num_samples),
               'mle_h4', 'pe', 'bucb', 'qEI', 'lp']

    method_names = [r'$\epsilon$-M-GPO  $H = 4$',
                    r'$\epsilon$-M-GPO  $H = 3$',
                    r'$\epsilon$-M-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$',  # r'MLE $H = 4$',
                    'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']

    # method_names = method_names[1:]
    # methods = methods[1:]

    output_file = root_path + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Branin,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Branin,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget,
                                         param_storer_string=param_storer_string)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    s = math.sqrt(428.88925324174573)
    # print (21.0982 - 18.6664) / s

    # print (3.9028 - 3.3118) / s

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
    num_samples = 50
    # camel
    # root_path = '../../tests/branin_new/camel_600_b{}_s{}/'.format(batch_size, num_samples)
    # seeds = range(35)
    # rewards = GetSimulatedTotalRewards(root_path=root_path,
    #                                    seeds=seeds,
    #                                    filename='camel_total_rewards.eps',
    #                                    param_storer_string='camel')
    # print
    # regrets = GetSimulatedTotalRegrets(root_path=root_path,
    #                                    seeds=seeds,
    #                                    filename='camel_simple_regrets.eps',
    #                                    param_storer_string= 'camel')
    # print
    # print

    # boha
    root_path = '../../tests/branin_new/boha_400_b{}_s{}/'.format(batch_size, num_samples)
    seeds = range(35, 70)
    rewards = GetSimulatedTotalRewards(root_path=root_path,
                                       seeds=seeds,
                                       filename='boha_total_rewards.eps',
                                       param_storer_string='boha')
    print
    regrets = GetSimulatedTotalRegrets(root_path=root_path,
                                       seeds=seeds,
                                       filename='boha_simple_regrets.eps',
                                       param_storer_string='boha')