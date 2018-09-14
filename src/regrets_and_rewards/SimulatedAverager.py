from src.enum.DatasetEnum import DatasetEnum
from src.enum.MetricsEnum import MetricsEnum
from src.metric.ResultCalculator import ResultCalculator
from src.plotting.ResultsPlotter import  ResultGraphPlotter


def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1',
               'mle_h4', 'new_fixed_pe', 'gp-bucb', 'qEI', 'my_lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI',
                    'BBO-LP']
    """
    seeds = range(66, 316)

    print(len(seeds))
    root_path = '../../tests/sim-fixed-temp/'
    methods = ['h4', 'h3', 'h2', 'h1', 'mle_h4', 'pe', 'bucb', 'qEI', 'lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']
    output_file = '../../result_graphs/eps/simulated/test_simulated_total_rewards.eps'

    result_calculator = ResultCalculator(dataset_type=DatasetEnum.Simulated,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds)
    results = result_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=MetricsEnum.AverageTotalReward)

    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Simulated,
                                         plotting_type=MetricsEnum.AverageTotalReward,
                                         batch_size=batch_size)
    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=False)

    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)


def GetSimulatedTotalRegrets():
    seeds = range(66, 102)

    batch_size = 4
    """
    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1', 'mle_h4', 'new_fixed_pe', 'gp-bucb', 'qEI', 'my_lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']
    """
    seeds = list(set(range(66, 331)) - set([152, 154, 175, 176, 327,
                                            296, 294, 293, 282, 278, 261, 254, 233, 220, 199, 195, 189,
                                            185, 183]))
    
    seeds = range(66, 316)

    print(len(seeds))
    root_path = '../../tests/sim-fixed-temp/'
    methods = ['h4','h3', 'h2', 'h1', 'mle_h4', 'pe', 'bucb', 'qEI', 'lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']

    output_file = '../../result_graphs/eps/simulated/test_simulated_simple_regrets.eps'

    regret_calculator = ResultCalculator(dataset_type=DatasetEnum.Simulated,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds)
    results = regret_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names,
                                                  metric_type=MetricsEnum.SimpleRegret)
    results_plotter = ResultGraphPlotter(dataset_type=DatasetEnum.Simulated,
                                         plotting_type=MetricsEnum.SimpleRegret,
                                         batch_size=batch_size)

    results_plotter.plot_results(results=results, output_file_name=output_file, plot_bars=False)
    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)


if __name__ == "__main__":
    GetSimulatedTotalRewards()
    GetSimulatedTotalRegrets()