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

    methods = ['anytime_h4',
               'anytime_h3',
               'anytime_h2',
               'h1',
               'mle_h4', 'new_ixed_pe', 'bucb', 'my_qEI', 'my_lp']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$',
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
                                input_folder='../../releaseTests/updated_release/road/new_new_new_beta2_c/',
                                filename='road_beta2_rewards.eps',
                                plot_bars=False)


def GetRoadBeta3Rewards():
    return CalculateMetricsBeta(h=3,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                input_folder='../../releaseTests/updated_release/road/new_beta3_c/',
                                filename='road_beta3_rewards.eps',
                                plot_bars=False)


def CalculateMetricsFull(metric_type,
                         plotting_type,
                         filename,
                         plot_bars):
    batch_size = 5
    total_budget = 20

    seeds = range(0, 35)

    root_path = '../../releaseTests/road/tests2full-r/'

    methods = ['anytime_h4', 'anytime_h2_full', 'anytime_h2', 'ei']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$  ($20$)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$  ($20$)',
                    'EI (all)']

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
    h2_all = results[1]
    h2 = results[2]
    print "H4 -  H2 all %f sigma " % round(abs((h4[1][-1] - h2_all[1][-1])) / sigma, 4)
    print "H2 all  -  H2 %f sigma" % round(abs((h2_all[1][-1] - h2[1][-1])) / sigma, 4)
    return results


def GetRoad_H2Full_TotalRewards():
    return CalculateMetricsFull(metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageTotalReward,
                                filename='road_h2_full_total_rewards.eps',
                                plot_bars=False)


def GetRoadTotalRegrets_H2Full():
    return CalculateMetricsFull(metric_type=MetricsEnum.SimpleRegret,
                                plotting_type=PlottingEnum.SimpleRegret,
                                filename='road_h2_full_simple_regrets.eps',
                                plot_bars=False)


def GetRoadBeta2Regrets():
    """
    seeds = range(35)
    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRegrets(batch_size, root_path, methods, method_names, seeds)
    """
    seeds = range(35)
    batch_size = 5
    """
    root_path = '../../road_tests/beta2/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    """
    root_path = '../../road_tests/new_new_new_beta2/'
    root_path = '../../temp/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    beta_list = [0.25]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names =  ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    """
    root_path = '../../road_tests/new_new_new_beta2/'
    # root_path = '../../temp/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)
    """
    """
    seed = range(42)
    root_path = '../../new_road_tests/beta2/'
    beta_list = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names =   map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    root_path = '../../releaseTests/updated_release/road/beta2r/'
    root_path = '../../road_tests/new_new_new_beta2/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)
    output_file = '../../result_graphs/eps/road/road_beta2_regrets.eps'
    """
    seed = range(42)
    root_path = '../../new_road_tests/beta2/'
    beta_list = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)
    """
    output_file = '../../result_graphs/eps/road/t_road_beta2_regrets.eps'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=MetricsEnum.SimpleRegret)


def GetRoadBeta3Regrets():
    seeds = range(35)
    batch_size = 5
    """
    root_path = '../../road_tests/beta3/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../road_tests/new_beta3/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/road/road_beta3_regrets.eps'

    seed = range(42)
    root_path = '../../new_road_tests/beta3n/'
    beta_list = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/road/t_road_beta3_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=MetricsEnum.SimpleRegret)


if __name__ == "__main__":
    # print(wrap_with_bucks(1.10002, 3.45555))
    """
    rewards = GetRoadTotalRewards()
    regrets = GetRoadTotalRegrets()
    print
    get_rewards_regrets_latex(rewards,regrets)
    
    print(" pi = {0:.2f} or {1:.4f}".format(7.11111, 1555.22222))
    beta2 = GetRoadBeta2Rewards()
    beta3 = GetRoadBeta3Rewards()
    print
    get_rewards_regrets_latex(beta2, beta3, process_beta=True)


    
    regrets_h2 = GetRoadTotalRegrets_H2Full()
    rewards_h2 = GetRoad_H2Full_TotalRewards()
    get_rewards_regrets_latex(rewards_h2, regrets_h2)
    """
    GetRoadTotalRewards()
    GetRoadTotalRegrets()
