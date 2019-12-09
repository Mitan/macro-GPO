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

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'
    root_path = '../../releaseTests/paper/road/rewards/'

    methods = ['anytime_h4',
               'anytime_h3',
               'anytime_h2',
               'h1',
               'mle_h4', 'new_ixed_pe', 'bucb', 'my_qEI', 'my_lp']

    method_names = [r'Anytime $\epsilon$-M-BO  $H = 4$',
                    r'Anytime $\epsilon$-M-BO  $H = 3$',
                    r'Anytime $\epsilon$-M-BO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI', 'BBO-LP']

    output_file = '../../result_graphs/eps/road/' + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Road,
                                         root_path=root_path,
                                         time_slot=18,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Road,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    sigma = math.sqrt(0.7486)
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    print "H4 -  H1 %f sigma " % round(abs((h1[1][-1] - h4[1][-1])) / sigma, 4)
    print "H4  -  MLE %f sigma" % round(abs((mle[1][-1] - h4[1][-1])) / sigma, 4)
    return results


def GetRoadTotalRewards():
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename='road_total_rewards.eps',
                            plot_bars=False)


def GetRoadTotalRegrets():
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename='road_simple_regrets.eps',
                            plot_bars=False)


def CalculateMetricsBeta(h, metric_type, filename, input_folder, plot_bars, plotting_type):
    batch_size = 5
    total_budget = 20

    seeds = range(0, 35)

    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = ['anytime_h' + str(h)] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/road/' + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Road,
                                         root_path=input_folder,
                                         time_slot=18,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Road,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    sigma = math.sqrt(0.7486)
    beta0 = results[0]
    beta02 = results[1]
    print "beta=0 -  beta=0.2 %f sigma " % round(abs((beta0[1][-1] - beta02[1][-1])) / sigma, 4)
    return results


def GetRoadBeta2Rewards():
    return CalculateMetricsBeta(h=2,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                input_folder='../../releaseTests/paper/road/beta2/',
                                filename='road_beta2_rewards.eps',
                                plot_bars=False)


def GetRoadBeta3Rewards():
    return CalculateMetricsBeta(h=3,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                input_folder='../../releaseTests/paper/road/beta3/',
                                filename='road_beta3_rewards.eps',
                                plot_bars=False)


def CalculateMetricsFull(metric_type,
                         plotting_type,
                         filename,
                         plot_bars):
    batch_size = 5
    total_budget = 20

    # seeds = list(set(range(0, 398)) -
    #              set(range(27,40) + range(58, 60) + [69] + range(71,80) + range(90,100) + range(111,120)
    #   + range(208, 220) + range(235, 240) + range(241, 260)
    # + range(267, 280) + range(293, 300) +
    #                  range(332, 334) + range(369, 370) + range(377, 378) + range(389, 390)
    #                  + range(396, 398)
    #                  ))
    seeds = range(35)
    print len(seeds)

    root_path = '../../releaseTests/road/tests2full-r/'
    root_path = '../../releaseTests/paper/road/tests2full/'
    # root_path = '../../tests/road_h2_b5_s300/'

    methods = ['anytime_h4', 'anytime_h2_full', 'anytime_h2', 'ei']

    method_names = [r'Anytime $\epsilon$-M-BO  $H = 4$  ($20$)',
                             r'Anytime $\epsilon$-M-BO  $H = 2$ (all)',
                             r'Anytime $\epsilon$-M-BO  $H = 2$  ($20$)',
                             'EI (all)']

    # methods = ['h2_b5_s300_all', 'h2_b5_s300_selected', 'ei']
    #
    # method_names = [
    #                          r'Anytime $\epsilon$-M-GPO  $H = 2$ (all)',
    #                          r'Anytime $\epsilon$-M-GPO  $H = 2$  ($20$)',
    #                          'EI (all)']
    # methods = methods[1:]
    # method_names = method_names[1:]

    output_file = root_path + filename

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Road,
                                         root_path=root_path,
                                         time_slot=18,
                                         seeds=seeds,
                                         total_budget=total_budget)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=metric_type)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Road,
                                         plotting_type=plotting_type,
                                         batch_size=batch_size,
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    # sigma = math.sqrt(0.7486)
    # h4 = results[0]
    # h2_all = results[1]
    # h2 = results[2]
    # print "H4 -  H2 all %f sigma " % round(abs((h4[1][-1] - h2_all[1][-1])) / sigma, 4)
    # print "H2 all  -  H2 %f sigma" % round(abs((h2_all[1][-1] - h2[1][-1])) / sigma, 4)
    return results


def GetRoad_H2Full_TotalRewards():
    return CalculateMetricsFull(metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardFull,
                                filename='road_h2_full_total_rewards.eps',
                                plot_bars=False)


def GetRoadTotalRegrets_H2Full():
    return CalculateMetricsFull(metric_type=MetricsEnum.SimpleRegret,
                                plotting_type=PlottingEnum.SimpleRegretFull,
                                filename='road_h2_full_simple_regrets.eps',
                                plot_bars=False)


if __name__ == "__main__":
    # GetRoadTotalRewards()
    # GetRoadTotalRegrets()
    beta2 = GetRoadBeta2Rewards()
    beta3 = GetRoadBeta3Rewards()
    #
    # regrets_h2 = GetRoadTotalRegrets_H2Full()
    # rewards_h2 = GetRoad_H2Full_TotalRewards()

