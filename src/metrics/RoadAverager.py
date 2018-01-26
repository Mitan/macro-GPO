from GeneralResultsAverager import RoadRewards
from src.PlottingEnum import PlottingMethods
from RegretCalculator import RoadRegrets


def GetRoadBeta2Rewards():
    seeds = range(35)
    # seeds = list(set(range(35))
    # root_path = '../../road_tests/beta2/'
    batch_size = 5
    """
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    root_path =  '../../road_tests/beta2/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta2/'
    # root_path = '../../road_tests/new_beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    """
    root_path = '../../road_tests/new_new_new_beta2/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names =  ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    root_path = '../../road_tests/new_beta225/'
    beta_list = [0.25]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    """
    root_path = '../../road_tests/beta2/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    """
    output_file = '../../result_graphs/eps/road/road_beta2_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)


def GetRoadBeta3Rewards():
    seeds = range(35)

    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    batch_size = 5
    """
    root_path = '../../road_tests/beta3/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta3/'
    """
    root_path = '../../road_tests/new_beta3/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    root_path = '../../road_tests/new_beta325/'
    beta_list = [0.25]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    output_file = '../../result_graphs/eps/road/road_beta3_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)


def GetRoad_H4Samples_TotalRewards():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h4_5', 'anytime_h4_50', 'anytime_h4']
    # methods = ['anytime_h4_5', 'anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    # method_names = [r'$N = 5$', r'$N = 300$']

    root_path = '../../road_tests/new_h4/'
    root_path = '../../releaseTests/road/h4_samples/rewards/'

    output_file = '../../result_graphs/eps/road_h4samples_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)

def GetRoadTotalRewards():
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    methods = ['h1', 'anytime_h2','anytime_h3', 'anytime_h4', 'mle_h4','fixed_pe', 'bucb', 'r_inf_qEI']
    # methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    # methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300']
    # methods = ['anytime_h4_300']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    # method_names = [ "H = 4"]

    root_path = '../../releaseTests/road/b5-18-log/'
    # root_path = '../../road_tests/tests1/'
    root_path = '../../new_road_tests/new_all_3/'

    output_file = '../../result_graphs/eps/temp_road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetRoad_H2Full_TotalRewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([22]))
    batch_size = 5

    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4', 'ei']
    # methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300', 'ei']
    # methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300']
    method_names = [ r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                     r'Anytime $\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                     r'Anytime $\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                     'EI (all MA)']

    root_path = '../../releaseTests/road/tests2full/'
    root_path = '../../releaseTests/road/tests2full-r/'
    # root_path = '../../releaseTests/road/tests2full-NEW/'

    output_file = '../../result_graphs/eps/road/road_h2_full_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


"""
def GetRoadBeta2Regrets():
    seeds = range(0, 35)
    # seeds = list(set(seeds) - set([27, 31]))
    # seeds = list(set(seeds) - set([5]))
    root_path = '../../road_tests/beta2/'
    root_path = '../../releaseTests/road/beta2/'

    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    # methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta2_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadBeta3Regrets():
    seeds = range(0, 35)
    # seeds = list(set(seeds) - set([27, 31]))
    # seeds = list(set(seeds) - set([5]))
    root_path = '../../road_tests/beta3/'

    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta3_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)
"""

def GetRoadTotalRegrets():
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'new_ixed_pe', 'bucb', 'r_qei']

    method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$', r'$H^* = 4$', r'MLE $H = 4$', 'GP-BUCB-PE', 'GP-BUCB',
                    'qEI']

    # methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300', 'mle_h4', 'new_ixed_pe', 'bucb', 'r_qei']
    # methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    methods = ['h1', 'anytime_h2', 'mle_h4', 'fixed_pe', 'bucb', 'r_inf_qEI']
    # methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    # methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300']
    # methods = ['anytime_h4_300']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI']

    # method_names = [ "H = 4"]

    root_path = '../../releaseTests/road/b5-18-log/'
    # root_path = '../../road_tests/tests1/'
    root_path = '../../new_road_tests/new_all_3/'

    output_file = '../../result_graphs/eps/temp_road_simple_regrets.eps'
    # root_path = '../../road_tests/tests1/'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets_H2Full():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300', 'ei']
    # methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300']
    methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4', 'ei']

    method_names = [ r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                     r'Anytime $\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                     r'Anytime $\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                     'EI (all MA)']

    output_file = '../../result_graphs/eps/road/road_h2_full_simple_regrets.eps'
    root_path = '../../releaseTests/road/tests2full-r/'
    # root_path = '../../releaseTests/road/tests2full-NEW/'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets_H4Samples():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h4_5', 'anytime_h4_50', 'anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    root_path = '../../releaseTests/road/h4_samples/rewards/'

    output_file = '../../result_graphs/eps/road_h4samples_simple_regrets.eps'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)



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
    root_path = '../../road_tests/new_new_new_beta2/'
    # root_path = '../../temp/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)

    output_file = '../../result_graphs/eps/road/road_beta2_regrets.eps'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


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
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/road/road_beta3_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)



if __name__ == "__main__":
    """
    GetRoadTotalRewards()
    GetRoadBeta3Rewards()
    GetRoadBeta2Rewards()

    # GetRoad_H2Full_TotalRewards()
    """
    #GetRoad_H4Samples_TotalRewards()
    # GetRoadTotalRegrets_H4Samples()
    # GetRoadBeta2Rewards()
    # GetRoadTotalRewards()
    # GetRoadTotalRewards()
    GetRoad_H2Full_TotalRewards()