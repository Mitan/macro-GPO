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
                     methods=None,
                     method_names=None):
    batch_size = 4
    total_budget = 20

    # seeds = range(66, 66 + 107)
    seeds = list(set(range(66, 66 + 109)) - set([116, 152, 154]))
    seeds = range(66,316)

    print(len(seeds))
    root_path = '../../tests/sim-fixed-temp/'
    if not methods:
        methods = ['h4', 'h3', 'h2', 'h1', 'mle_h4', 'pe', 'bucb', 'qEI', 'lp_1']

    if not method_names:
        method_names = [r'$\epsilon$-M-GPO  $H = 4$',
                        r'$\epsilon$-M-GPO  $H = 3$',
                        r'$\epsilon$-M-GPO  $H = 2$',
                        'DB-GP-UCB',
                        r'Nonmyopic GP-UCB $H = 4$', # r'MLE $H = 4$',
                        'GP-UCB-PE', 'GP-BUCB',
                        r'$q$-EI', 'BBO-LP']
    output_file = '../../result_graphs/eps/simulated/' + filename

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
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)
    """
    print round(results[0][1][-1] - results[3][1][-1], 4)
    print round(results[0][1][-1] - results[4][1][-1], 4)
    """
    return results


def CalculateMetricsBeta(h, metric_type, filename, plot_bars, plotting_type):
    batch_size = 4
    total_budget = 20

    seeds = range(66, 316)

    # root_path = '../../tests/beta%d_t/' % h
    root_path = '../../tests/beta%d/' % h

    beta_list = [0.0, 0.05, 0.15, 0.3, 0.5, 1.0, 2.0, 5.0]
    # beta_list = [0.0, 0.05, 0.1]

    methods = map(lambda x: 'beta' + str(x), beta_list)
    method_names = map(lambda x: 'beta = ' + str(2 * x), beta_list)

    output_file = '../../result_graphs/eps/simulated/' + filename

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
                                         total_budget=total_budget)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=plot_bars)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)

    print round(results[1][1][-1] - results[0][1][-1], 4)
    return results


def GetSimulatedTotalRewards():
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename='simulated_total_rewards.eps',
                            plot_bars=False)


def GetSimulatedTotalRegrets():
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename='simulated_simple_regrets.eps',
                            plot_bars=False)


def AverageRewardsBeta2():
    return CalculateMetricsBeta(h=2,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                filename='simulated_beta2_rewards.eps',
                                plot_bars=False)


def SimpleRegretBeta2():
    CalculateMetricsBeta(h=2,
                         metric_type=MetricsEnum.SimpleRegret,

                         filename='simulated_beta2_regrets.eps',
                         plot_bars=False)


def AverageRewardsBeta3():
    return CalculateMetricsBeta(h=3,
                                metric_type=MetricsEnum.AverageTotalReward,
                                plotting_type=PlottingEnum.AverageRewardBeta,
                                filename='simulated_beta3_rewards.eps',
                                plot_bars=False)


def SimpleRegretBeta3():
    CalculateMetricsBeta(h=3,
                         metric_type=MetricsEnum.SimpleRegret,
                         filename='simulated_beta3_regrets.eps',
                         plot_bars=False)


def GetSimulatedTotalRewardsRollout():
    methods = ['h4_b1_20', 'rollout_h4_gamma1_ei_mod']
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', 'Rollout-$4$-$10$']
    return CalculateMetrics(metric_type=MetricsEnum.AverageTotalReward,
                            plotting_type=PlottingEnum.AverageTotalReward,
                            filename='simulated_total_rewards_rollout.eps',
                            plot_bars=True,
                            methods=methods,
                            method_names=method_names)


def GetSimulatedTotalRegretsRollout():
    methods = ['h4_b1_20', 'rollout_h4_gamma1_ei_mod']
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', 'Rollout-$4$-$10$']
    return CalculateMetrics(metric_type=MetricsEnum.SimpleRegret,
                            plotting_type=PlottingEnum.SimpleRegret,
                            filename='simulated_simple_regrets_rollout.eps',
                            plot_bars=True,
                            methods=methods,
                            method_names=method_names)


if __name__ == "__main__":
    """
    rewards = GetSimulatedTotalRewards()
    regrets = GetSimulatedTotalRegrets()
    print
    get_rewards_regrets_latex(rewards, regrets)
    
    beta2 = AverageRewardsBeta2()
    beta3 = AverageRewardsBeta3()
    print
    print
    get_rewards_regrets_latex(beta2, beta3, process_beta=True)
   
    GetSimulatedTotalRewardsRollout()
    GetSimulatedTotalRegretsRollout()
    
    AverageRewardsBeta2()
    AverageRewardsBeta3()
    """
    rewards = GetSimulatedTotalRewards()
    regrets = GetSimulatedTotalRegrets()
    # AverageRewardsBeta2()
    # AverageRewardsBeta3()
