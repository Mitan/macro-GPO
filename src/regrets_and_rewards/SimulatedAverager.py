from GeneralResultsAverager import SimulatedRewards, SimulatedCumulativeRegrets
from src.enum.DatasetEnum import DatasetEnum
from src.enum.PlottingEnum import PlottingMethods
from src.metric.RegretCalculator import RegretCalculator
from src.plotting.ResultsPlotter import PlotData


def GetSimulatedTotalRewards(my_ei=True):
    if my_ei:
        ei_method = 'qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(66, 102)
    batch_size = 4

    # root_path = '../../releaseTests/simulated/rewards-sAD/'
    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'
    # root_path = '../../releaseTests/simulated/rewards-sAD-qei/'

    methods = ['h4', 'h3', 'h2', 'h1',
               'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method, 'my_lp']
    # 'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method,  'bbo-llp22']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI',
                    'BBO-LP']

    output_file = '../../result_graphs/eps/simulated/' + ei_folder + '/simulated_total_rewards.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/simulated_total_rewards.eps'

    results = SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                               method_names=method_names,
                               seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    # print "Rewards H4 / H1 %f" % (h4[1][-1] / h1[1][-1])
    # print "Rewards H4 / MLE %f" % (h4[1][-1]/ mle[1][-1])
    print results


def GetSimulatedTotalRewards_onlyH4(my_ei=True):
    if my_ei:
        ei_method = 'qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'
    # root_path = '../../releaseTests/simulated/rewards-sAD-qei/'

    methods = ['h4', 'h1', 'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method, 'bbo-llp4']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']

    output_file = '../../result_graphs/eps/simulated/' + ei_folder + '/h4_simulated_total_rewards.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/h4_simulated_total_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetSimulatedTotalRewards_our():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../releaseTests/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB']

    output_file = '../../result_graphs/eps/simulated/our_simulated_total_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetSimulatedBeta2Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    root_path = '../../simulated_tests/beta2-good/'

    beta_list = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta2_rewards.eps'

    results = SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                               method_names=method_names,
                               seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)
    beta0 = results[0]
    beta02 = results[1]

    print "Rewards beta0.2 / beta0.0 %f" % (beta02[1][-1] / beta0[1][-1])


def GetSimulatedBeta3Rewards():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    """
    root_path = '../../simulated_tests/beta3-good/'
    # root_path = '../../simulated_tests/beta3n/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    # beta_list = [0.25, 0.1]

    str_beta = map(str, beta_list)
    methods = ['h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta3_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)


def GetSimulatedTotalRewards_B1():
    seeds = range(66, 102)
    seeds = list(set(range(66, 175)) - set([154, 152]))
    batch_size = 4
    batch_size = 1

    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'
    root_path = '../../sim-fixed-temp/'

    methods = ['h4_b1_20', 'rollout_h4_gamma1', 'rollout_h4_q3_pi']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Rollout-$4$-$10$,  $q=1$',
                    r'Rollout-$4$-$10$,  $q=3$']

    methods = ['h4_b1_20', 'rollout_h4_gamma1_ei']
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Rollout-$4$-$10$']

    output_file = '../../result_graphs/eps/simulated/' + 'my_ei' + '/h4_simulated_total_rewards_rollout_b1.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/h4_simulated_total_rewards.eps'

    results = SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                               method_names=method_names,
                               seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)
    # print results
    print results[1][1][-1], results[0][1][-1]
    print results[1][2][-1], results[0][2][-1]


def GetSimulatedTotalRewards_B1_CUMULATIVE():
    seeds = range(66, 102)
    seeds = list(set(range(66, 175)) - set([154, 152]))
    batch_size = 4
    batch_size = 1

    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'
    root_path = '../../sim-fixed-temp/'

    methods = ['h4_b1_20', 'rollout_h4_gamma1', 'rollout_h4_q3_pi']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Rollout-$4$-$10$,  $q=1$',
                    r'Rollout-$4$-$10$,  $q=3$']

    methods = ['h4_b1_20', 'rollout_h4_gamma1_ei_mod']
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Rollout-$4$-$10$']

    output_file = '../../result_graphs/eps/simulated/' + 'my_ei' + '/h4_simulated_total_rewards_rollout_b1_cumulative.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/h4_simulated_total_rewards.eps'

    results = SimulatedCumulativeRegrets(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                                         method_names=method_names,
                                         seeds=seeds, output_filename=output_file,
                                         plottingType=PlottingMethods.CumulativeRegret)
    # print results
    print results[0][1][-1], results[1][1][-1]
    print results[0][2][-1], results[1][2][-1]
    print results


def GetSimulatedTotalRegrets():
    seeds = range(66, 102)

    batch_size = 4

    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1', 'mle_h4', 'new_fixed_pe', 'gp-bucb', 'qEI', 'my_lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']
    """
    seeds = list(set(range(66, 175)) - set([154, 152]))
    root_path = '../../sim-fixed-temp/'
    methods = ['h4', 'h1', 'mle_h4', 'pe', 'bucb', 'qEI', 'lp']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']
    """
    output_file = '../../result_graphs/eps/simulated/simulated_simple_regrets.eps'

    regret_calculator = RegretCalculator(dataset_type=DatasetEnum.Simulated,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds)
    results = regret_calculator.process_regrets(batch_size=batch_size,
                                                methods=methods,
                                                method_names=method_names)
    PlotData(results=results, output_file_name=output_file,
             plottingType=PlottingMethods.SimpleRegret, dataset=DatasetEnum.Simulated, plot_bars=True)
    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)


def GetSimulatedTotalRegrets_B1():
    seeds = list(set(range(66, 175)) - set([154, 152]))
    batch_size = 1

    root_path = '../../sim-fixed-temp/'

    methods = ['h4_b1_20', 'rollout_h4_gamma1_ei_mod']
    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Rollout-$4$-$10$']

    output_file = '../../result_graphs/eps/simulated/simulated_simple_regrets_rollout_b1.eps'

    regret_calculator = RegretCalculator(dataset_type=DatasetEnum.Simulated,
                                         root_path=root_path,
                                         time_slot=None,
                                         seeds=seeds)
    results = regret_calculator.process_regrets(batch_size=batch_size,
                                                methods=methods,
                                                method_names=method_names)
    PlotData(results=results, output_file_name=output_file,
             plottingType=PlottingMethods.SimpleRegret, dataset=DatasetEnum.Simulated, plot_bars=True)

    print results[1][1][-1], results[0][1][-1]
    print results[1][2][-1], results[0][2][-1]


if __name__ == "__main__":
    GetSimulatedTotalRegrets()
    GetSimulatedTotalRegrets_B1()

