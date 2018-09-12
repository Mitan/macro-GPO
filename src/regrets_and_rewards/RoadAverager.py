import math

from GeneralResultsAverager import RoadRewards
from src.enum.DatasetEnum import DatasetEnum
from src.enum.PlottingEnum import PlottingMethods
from src.metric.ResultCalculator import ResultCalculator
from src.plotting.ResultsPlotter import PlotData


def GetRoadBeta2Rewards():
    seeds = range(35)
    # seeds = range(42)
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
    # root_path = '../../releaseTests/updated_release/road/new_new_new_beta2/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)
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
    
    root_path = '../../new_road_tests/beta2/'
    beta_list = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0,  5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names =   map(lambda x: 'beta = ' + str(2* float(x)), str_beta)
    """
    output_file = '../../result_graphs/eps/road/road_beta2_rewards.eps'
    # output_file = '../../result_graphs/eps/road/n_road_beta2_rewards.eps'

    results = RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                          method_names=method_names,
                          seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalRewardBeta)
    beta0 = results[0]
    beta02 = results[1]
    # print beta0, beta02

    print "Rewards beta0.2 / beta0.0 %f" % (beta02[1][-1] / beta0[1][-1])


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
    # root_path = '../../releaseTests/updated_release/road/new_beta3/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2 * float(x)), str_beta)
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


def GetRoadTotalRewards(my_ei=True):
    if my_ei:
        ei_method = 'my_qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(35)

    batch_size = 5

    # methods = [ 'anytime_h4', 'anytime_h3', 'anytime_h2', 'h1', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    methods = ['anytime_h4', 'anytime_h3', 'anytime_h2', 'h1',
               'mle_h4', 'new_ixed_pe', 'bucb', ei_method, 'my_lp']
    #  'mle_h4','new_ixed_pe', 'bucb', ei_method,'bbo-llp7']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$', 'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI', 'BBO-LP']

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road/' + ei_folder + '/road_total_rewards.eps'
    # output_file = '../../result_graphs/eps/road/my_ei/road_total_rewards.eps'

    results = RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                          method_names=method_names,
                          seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)
    """
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    # print h4, h1, mle
    print "Rewards H4 / H1 %f" % (h4[1][-1] / h1[1][-1])
    print "Rewards H4 / MLE %f" % (h4[1][-1] / mle[1][-1])
    """
    print results


def GetRoadTotalRewards_ours():
    seeds = range(35)

    batch_size = 5

    methods = ['anytime_h4', 'anytime_h3', 'anytime_h2', 'h1']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$', 'DB-GP-UCB']

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road/ours_road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetRoadTotalRewards_ours_ucb():
    seeds = range(35)

    batch_size = 5

    methods = ['anytime_h4', 'anytime_h3', 'anytime_h2', 'h1', 'mle_h4', 'new_mle_h3', 'new_mle_h2']

    method_names = [r'$\epsilon$-Macro-GPO  $H = 4$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB', r'Nonmyopic GP-UCB $H = 4$', r'Nonmyopic GP-UCB $H = 3$', r'Nonmyopic GP-UCB $H = 2$']

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road/ucb_ours_road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetRoadTotalRewards_onlyH4(my_ei=True):
    if my_ei:
        ei_method = 'my_qEI'
        ei_folder = 'my_ei'
    else:
        ei_method = 'r_qei'
        ei_folder = 'r_ei'

    seeds = range(35)

    batch_size = 5

    # methods = [ 'anytime_h4',  'h1', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    methods = ['anytime_h4', 'h1', 'mle_h4', 'new_ixed_pe', 'bucb', ei_method, 'bbo-llp4']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$', 'DB-GP-UCB',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI', 'BBO-LP']

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road/' + ei_folder + '/onlyh4_road_total_rewards.eps'
    # output_file = '../../result_graphs/eps/road/my_ei/onlyh4_road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)


def GetRoad_H2Full_TotalRewards():
    seeds = range(35)
    batch_size = 5

    # methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    methods = ['anytime_h4', 'anytime_h2_full', 'anytime_h2', 'ei']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$  ($20$)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$  ($20$)',
                    'EI (all)']

    root_path = '../../releaseTests/road/tests2full/'
    root_path = '../../releaseTests/road/tests2full-r/'

    output_file = '../../result_graphs/eps/road/road_h2_full_total_rewards.eps'

    results = RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods,
                          method_names=method_names,
                          seeds=seeds, output_filename=output_file, plottingType=PlottingMethods.TotalReward)
    # print results
    h4 = results[0]
    h2_all = results[1]
    h2 = results[2]
    # print h4, h2_all, h2
    print "Rewards H4 / H2 all %f" % (h4[1][-1] / h2_all[1][-1])
    print "Rewards H2  / H2 all %f" % (1 - h2[1][-1] / h2_all[1][-1])


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
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets():
    seeds = range(35)
    batch_size = 5

    time_slot = 18

    methods = ['anytime_h4', 'anytime_h3', 'anytime_h2', 'h1',
               'mle_h4', 'new_ixed_pe', 'bucb', 'my_qEI', 'my_lp']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$',
                    'DB-GP-UCB',
                    r'Nonmyopic GP-UCB $H = 4$',
                    'GP-UCB-PE', 'GP-BUCB', r'$q$-EI', 'BBO-LP']

    root_path = '../../releaseTests/updated_release/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road/road_simple_regrets.eps'

    regret_calculator = ResultCalculator(dataset_type=DatasetEnum.Road,
                                         root_path=root_path,
                                         time_slot=time_slot,
                                         seeds=seeds)
    results = regret_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names)
    PlotData(results=results, output_file_name=output_file,
             plottingType=PlottingMethods.SimpleRegret, dataset=DatasetEnum.Road, plot_bars=True)
    """
    sigma = math.sqrt(0.7486)
    h4 = results[0]
    h1 = results[3]
    mle = results[4]
    print "Regrets H4 -  H1 %f sigma " % ((h1[1][-1] - h4[1][-1]) / sigma)
    print "Regrets H4  -  MLE %f sigma" % ((mle[1][-1] - h4[1][-1]) / sigma)
    """
    for result in results:
        print result[0], round(result[1][-1], 4), '+-', round(result[2][-1], 4)


def GetRoadTotalRegrets_H2Full():
    seeds = range(35)
    batch_size = 5
    time_slot = 18

    methods = ['anytime_h4', 'anytime_h2_full', 'anytime_h2', 'ei']

    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 4$  ($20$)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$  ($20$)',
                    'EI (all)']

    output_file = '../../result_graphs/eps/road/road_h2_full_simple_regrets.eps'
    root_path = '../../releaseTests/road/tests2full-r/'
    # root_path = '../../releaseTests/road/tests2full-NEW/'

    regret_calculator = ResultCalculator(dataset_type=DatasetEnum.Road,
                                         root_path=root_path,
                                         time_slot=time_slot,
                                         seeds=seeds)
    results = regret_calculator.calculate_results(batch_size=batch_size,
                                                  methods=methods,
                                                  method_names=method_names)
    PlotData(results=results, output_file_name=output_file,
             plottingType=PlottingMethods.SimpleRegret, dataset=DatasetEnum.Road, plot_bars=True)
    """
    h4 = results[0]
    h2_all = results[1]
    h2 = results[2]
    # print h4, h2_all, h2
    sigma = math.sqrt(0.7486)

    print "Regrets H4 -  H2 all %f sigma " % ((h2_all[1][-1] - h4[1][-1]) / sigma)
    print "Regrets H2 all  -  H2 %f sigma" % ((h2[1][-1] - h2_all[1][-1]) / sigma)
    """


if __name__ == "__main__":
    GetRoadTotalRegrets()
    GetRoadTotalRegrets_H2Full()
