from GeneralResultsAverager import SimulatedRewards
from src.PlottingEnum import PlottingMethods
from RegretCalculator import SimulatedRegrets


def GetSimulatedTotalRewards(my_ei = True):
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
               'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method]
               # 'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method,  'bbo-llp4']

    method_names = [ r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                     'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI',
                     'BBO-LP']
    
    output_file = '../../result_graphs/eps/simulated/' + ei_folder + '/simulated_total_rewards.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/simulated_total_rewards.eps'

    results = SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    print "Rewards H4 / H1 %f" % (h4[1][-1] / h1[1][-1])
    print "Rewards H4 / MLE %f" % (h4[1][-1]/ mle[1][-1])

def GetSimulatedTotalRewards_onlyH4(my_ei = True):
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


def GetSimulatedTotalRegrets(my_ei = True):
    if my_ei:
        ei_method = 'qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(66, 102)
    batch_size = 4

    # root_path = '../../releaseTests/simulated/rewards-sAD/'
        # root_path = '../../releaseTests/simulated/rewards-sAD-qei/'

    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1',
               'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method]
               # 'bbo-llp4']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                     r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI',
                    'BBO-LP']

    output_file = '../../result_graphs/eps/simulated/'+ ei_folder + '/simulated_simple_regrets.eps'
    # output_file = '../../result_graphs/eps/simulated/my_ei/simulated_simple_regrets.eps'

    results = SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)
    # print results
    sigma = 1.0
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    # print h4, h1, mle

    """
    print h4
    print h1
    print mle
    """
    print "Regrets H4 -  H1 %f sigma " % ((h1[1][-1] - h4[1][-1]) / sigma)
    print "Regrets H4  -  MLE %f sigma" % ((mle[1][-1] - h4[1][-1]) / sigma)


def GetSimulatedTotalRegrets_onlyH4(my_ei= True):
    if my_ei:
        ei_method = 'qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(66, 102)
    batch_size = 4

    # root_path = '../../releaseTests/simulated/rewards-sAD/'
    # root_path = '../../releaseTests/simulated/rewards-sAD-qei/'
    root_path = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    methods = ['h4', 'h1', 'mle_h4', 'new_fixed_pe', 'gp-bucb', ei_method, 'bbo-llp4']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$',
                    'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI', 'BBO-LP']
    # output_file = '../../result_graphs/eps/simulated/my_ei/h4_simulated_simple_regrets.eps'
    output_file = '../../result_graphs/eps/simulated/' + ei_folder + '/h4_simulated_simple_regrets.eps'

    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedTotalRegrets_our():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../releaseTests/simulated/rewards-sAD/'

    methods = ['h4', 'h3', 'h2', 'h1']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB']

    output_file = '../../result_graphs/eps/simulated/our_simulated_simple_regrets.eps'

    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


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
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' +  str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta2_rewards.eps'

    results = SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
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
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta3_rewards.eps'

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
    pass
    # GetSimulatedBeta2Rewards()
    # GetSimulatedBeta3Rewards()
    # GetSimulatedTotalRewards()

    # GetSimulated_H4Samples_TotalRewards()