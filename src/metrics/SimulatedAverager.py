from GeneralResultsAverager import SimulatedRewards
from src.PlottingEnum import PlottingMethods


def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', '2_s250_100k_anytime_h4', 'mle_h4', 'new_fixed_pe', 'gp-bucb', 'r_qei']

    method_names = ['DB-GP-UCB', r'$\epsilon$-Macro-GPO  $H = 2$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI']

    output_file = '../../result_graphs/eps/simulated_total_rewards.eps'

    """
    root_path = '../../simulated_tests/anytime/'
    methods = ['anytime_h4_300']
    method_names = ['Anytime']
    output_file = '../../result_graphs/eps/temp_simulated_total_rewards.eps'
    """
    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetSimulatedBeta2Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/simulated_beta2_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)


def GetSimulatedBeta3Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/simulated_beta3_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)


def GetSimulated_H4Samples_TotalRewards():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../simulated_tests/h4_samples/'
    root_path = '../../releaseTests/simulated/h4_samples/'
    methods = ['h4', 'new_new_h4_20', 'h4_5']
    method_names = ['N=100', 'N=20','N=5']
    output_file = '../../result_graphs/eps/simulated_h4_samples_total_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


if __name__ == "__main__":

    GetSimulatedBeta2Rewards()
    GetSimulatedBeta3Rewards()
    GetSimulatedTotalRewards()

    GetSimulated_H4Samples_TotalRewards()