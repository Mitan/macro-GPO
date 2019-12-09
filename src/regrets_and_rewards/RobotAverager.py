import math

from src.Utils import get_rewards_regrets_latex
from src.enum.PlottingEnum import PlottingEnum
from src.enum.DatasetEnum import DatasetEnum
from src.enum.MetricsEnum import MetricsEnum
from src.metric.ResultCalculator import ResultCalculator
from src.plotting.ResultsPlotter import ResultGraphPlotter


def CalculateMetrics(metric_type,
                     plotting_type,
                     filename,
                     plot_bars):
    batch_size = 5
    total_budget = 20

    seeds = range(0, 35)
    time_slot = 16
    dataset_type = DatasetEnum.Robot

    root_path = '../../releaseTests/updated_release/robot/all_tests_release/'
    root_path = '../../releaseTests/paper/robot/all_tests_release/'

    methods = ['new_anytime_h4_300',
               'anytime_h3',
               'anytime_h2',
               'anytime_h1',
               'mle_h4',
               'pe', 'bucb', 'my_qEI', 'lp']

    method_names = [r'Anytime $\epsilon$-M-BO  $H = 4$',
                    r'Anytime $\epsilon$-M-BO  $H = 3$',
                    r'Anytime $\epsilon$-M-BO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$',
                    'GP-UCB-PE', 'GP-BUCB', r'$q$-EI', 'BBO-LP']

    output_file = '../../result_graphs/eps/robot/' + filename

    result_calculator = ResultCalculator(dataset_type=dataset_type,
                                         root_path=root_path,
                                         time_slot=time_slot,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=dataset_type,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    sigma = math.sqrt(0.596355)
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    # print h4, h1, mle
    print "H4 -  H1 %f sigma " % abs((h1[1][-1] - h4[1][-1]) / sigma)
    print "H4  -  MLE %f sigma" % abs((mle[1][-1] - h4[1][-1]) / sigma)
    return results


def GetRobotTotalRegrets():
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename='robot_simple_regrets.eps',
                            plot_bars=False)


def GetRobotTotalRewards():
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename='robot_total_rewards.eps',
                            plot_bars=False)


def CalculateMetricsBeta(h, metric_type, filename, input_folder, plot_bars, plotting_type):
    batch_size = 5
    total_budget = 20

    seeds = range(0, 35)
    time_slot = 16
    dataset_type = DatasetEnum.Robot

    # root_path = '../../noise_robot_tests/release/beta2_release/'

    beta_list = [0.5, 1.0, 1.5, 2.0, 5.0]
    beta_list = [0.5, 1.0, 1.5, 2.0]

    str_beta = map(str, beta_list)
    methods = ['anytime_h' + str(h)] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/robot/' + filename

    result_calculator = ResultCalculator(dataset_type=dataset_type,
                                         root_path=input_folder,
                                         time_slot=time_slot,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=dataset_type,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    sigma = math.sqrt(0.596355)
    beta0 = results[0]
    beta1 = results[1]
    print "beta=0 -  beta=1.0 %f sigma " % abs((beta0[1][-1] - beta1[1][-1]) / sigma)
    return results


def GetRobotBeta2Rewards():
    return CalculateMetricsBeta(h=2,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                input_folder='../../releaseTests/paper/robot/beta2/',
                                filename='robot_beta2_rewards.eps',
                                plot_bars=False)


def GetRobotBeta3Rewards():
    return CalculateMetricsBeta(h=3,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                input_folder='../../releaseTests/paper/robot/beta3/',
                                filename='robot_beta3_rewards.eps',
                                plot_bars=False)


def GetRobot_H2Full_TotalRewards():
    return CalculateMetricsFull(metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardFull,
                                filename='robot_h2_full_total_rewards.eps',
                                plot_bars=False)


def CalculateMetricsFull(metric_type,
                         plotting_type,
                         filename,
                         plot_bars):
    batch_size = 5
    total_budget = 20

    seeds = range(0, 35)
    time_slot = 16
    dataset_type = DatasetEnum.Robot

    methods = ['new_anytime_h4_300', 'anytime_h2_full', 'anytime_h2', 'ei']

    method_names = [r'Anytime $\epsilon$-M-BO  $H = 4$  ($20$)',
                    r'Anytime $\epsilon$-M-BO  $H = 2$ (all)',
                    r'Anytime $\epsilon$-M-BO  $H = 2$  ($20$)',
                    'EI (all)']

    # root_path = '../../noise_robot_tests/release/all_tests_release/'
    root_path = '../../releaseTests/updated_release/robot/all_tests_release/'
    root_path = '../../releaseTests/paper/robot/all_tests_release/'

    output_file = '../../result_graphs/eps/robot/' + filename

    result_calculator = ResultCalculator(dataset_type=dataset_type,
                                         root_path=root_path,
                                         time_slot=time_slot,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=dataset_type,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    sigma = math.sqrt(0.596355)
    h4 = results[0]
    h2_all = results[1]
    h2 = results[2]
    print "H4 -  H2 all %f sigma " % round(abs((h4[1][-1] - h2_all[1][-1])) / sigma, 4)
    print "H2 all  -  H2 %f sigma" % round(abs((h2_all[1][-1] - h2[1][-1])) / sigma, 4)
    return results


def GetRobotTotalRegrets_H2Full():
    return CalculateMetricsFull(metric_type=MetricsEnum.SimpleRegret,
                                plotting_type=PlottingEnum.SimpleRegretFull,
                                filename='robot_h2_full_simple_regrets.eps',
                                plot_bars=False)




if __name__ == "__main__":

    # regrets_h2 = GetRobotTotalRegrets_H2Full()
    # rewards_h2 = GetRobot_H2Full_TotalRewards()
    #
    # regrets = GetRobotTotalRegrets()
    # rewards = GetRobotTotalRewards()

    beta2 = GetRobotBeta2Rewards()
    beta3 = GetRobotBeta3Rewards()