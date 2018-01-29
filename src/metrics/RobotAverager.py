from GeneralResultsAverager import RobotRewards
from src.PlottingEnum import PlottingMethods
from RegretCalculator import RobotRegrets


def GetRobotBeta2Rewards():
    seeds = range(35)
    seeds = range(35)
    time_slot = 16
    batch_size = 5
    """
    root_path = '../../releaseTests/robot/beta_new2/'
    root_path = '../../noise_robot_tests/beta2_fixed_exp/'
    # root_path = '../../noise_robot_tests/beta2/'
    root_path = '../../noise_robot_tests/beta222/'
    # root_path = '../../noise_robot_tests/beta2_fixed/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.05, 0.1,  0.5, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]
    batch_size = 5
    time_slot = 16

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    beta_list = [0.5, 1.0, 1.5, 2.0, 5.0]
    root_path = '../../noise_robot_tests/release/beta2_release/'
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2*float(x)), str_beta)

    output_file = '../../result_graphs/eps/noise_robot_beta2_rewards.eps'
    output_file = '../../result_graphs/eps/additional/noise_robot_beta2_rewards.eps'
    output_file = '../../result_graphs/eps/robot_beta2_rewards.eps'
    output_file = '../../result_graphs/eps/robot/robot_beta2_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot,
                 plottingType=PlottingMethods.TotalRewardBeta)


def GetRobotBeta3Rewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([33, 34]))
    root_path = '../../releaseTests/robot/beta_new3/'
    # root_path = '../../robot_tests/beta3/'
    # root_path = '../../noise_robot_tests/beta3_fixed_exp/'
    root_path = '../../noise_robot_tests/beta3/'
    root_path = '../../noise_robot_tests/release/beta3_release-r/'
    """
    # root_path = '../../noise_robot_tests/new_beta1532/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]
    # beta_list = [0.0,  0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 0.1, 1.0, 2.0, 5.0]

    # beta_list = [1.5]

    # beta_list = [0.0, 0.05, 0.1]
    """
    batch_size = 5
    time_slot = 16

    beta_list = [0.5, 1.0, 1.5, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2*float(x)), str_beta)
    """
    output_file = '../../result_graphs/eps/robot_beta3_rewards.eps'
    output_file = '../../result_graphs/eps/noise_robot_beta3_rewards.eps'
    output_file = '../../result_graphs/eps/additional/noise_robot_beta3_rewards.eps'
    """
    output_file = '../../result_graphs/eps/robot/robot_beta3_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot,
                 plottingType=PlottingMethods.TotalRewardBeta)


def GetRobot_H4Samples_TotalRewards():
    seeds = range(35)
    # seeds = list(set(range(35))
    batch_size = 5

    time_slot = 16

    # methods = ['new_anytime_h4_ 5', 'anytime_h4_ 50', 'anytime_h4']
    # methods = ['new_anytime_h4_ 5', 'anytime_h4_5', 'anytime_h4']
    # methods = ['new_anytime_h4_ 5', 'anytime_h4_5']
    methods = ['new_anytime_h4_5', 'new_anytime_h4_50', 'new_anytime_h4_300']
    # methods = ['n_anytime_h4_5','new_anytime_h4_5', 'new_anytime_h4_50', 'new_anytime_h4_300']
    # methods = ['new_anytime_h4_5']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    # method_names = [r'$N = 5$a', r'$N = 5$', r'$N = 50$', r'$N = 300$']

    # root_path = '../../robot_tests/h4_samples/'
    root_path = '../../noise_robot_tests/h4_tests/'
    root_path = '../../releaseTests/noised_robot/h4_tests/'

    # output_file = '../../result_graphs/eps/robot_h4samples_total_rewards.eps'
    output_file = '../../result_graphs/eps/noise_robot_h4samples_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, plottingType=PlottingMethods.TotalReward)


def GetRobotTotalRewards():
    seeds = range(35)

    batch_size = 5

    time_slot = 16

    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'new_anytime_h4_300', 'mle_h4', 'r_qei', 'pe', 'gp-bucb']
    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'mle_h4', 'pe', 'gp-bucb', 'qEI']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'MLE $H = 4$',  'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    root_path = '../../a_robot/all_tests/'

    output_file = '../../result_graphs/eps/a_robot/robot_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot,
                 plottingType=PlottingMethods.TotalReward)


def GetRobot_H2Full_TotalRewards():
    seeds = range(35)
    batch_size = 5
    time_slot = 16

    # methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4']
    # methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4', 'ei']
    methods = ['anytime_h2_full', 'anytime_h2', 'new_anytime_h4_300', 'ei']

    # methods = ['anytime_h2_full']

    # method_names = [r'$H^* = 2$ (all MA)', r'$H^* = 2$ (selected MA)', r'$H^* = 4$ (selected MA)']
    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                    'EI (all MA)']

    root_path = '../../noise_robot_tests/all_tests/'
    root_path = '../../noise_robot_tests/release/all_tests_release/'

    output_file = '../../result_graphs/eps/robot_h2_full_total_rewards.eps'
    output_file = '../../result_graphs/eps/robot/robot_h2_full_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, plottingType=PlottingMethods.TotalReward)


def GetRobotTotalRegrets():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'new_anytime_h4_300', 'mle_h4', 'r_qei', 'pe', 'gp-bucb']
    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'mle_h4', 'pe', 'gp-bucb', 'qEI']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']

    method_names = ['DB-GP-UCB', r'Anytime $\epsilon$-Macro-GPO  $H = 2$', r'Anytime $\epsilon$-Macro-GPO  $H = 3$',
                    r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    root_path = '../../a_robot/all_tests/'

    output_file = '../../result_graphs/eps/a_robot/robot_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_H2Full():
    seeds = range(35)
    batch_size = 5
    time_slot = 16

    methods = ['anytime_h2_full', 'anytime_h2', 'new_anytime_h4_300', 'ei']

    # methods = ['anytime_h2_full']

    # method_names = [r'$H^* = 2$ (all MA)', r'$H^* = 2$ (selected MA)', r'$H^* = 4$ (selected MA)']
    method_names = [r'Anytime $\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                    r'Anytime $\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                    'EI (all MA)']

    root_path = '../../noise_robot_tests/all_tests/'
    root_path = '../../noise_robot_tests/release/all_tests_release/'

    output_file = '../../result_graphs/eps/robot/robot_h2_full_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_H4Samples():
    seeds = range(35)
    batch_size = 5

    methods = ['new_anytime_h4_5', 'new_anytime_h4_50', 'new_anytime_h4_300']

    method_names = [r'$N = 5$',r'$N = 50$', r'$N = 300$']

    root_path = '../../noise_robot_tests/h4_tests/'
    root_path = '../../releaseTests/noised_robot/h4_tests/'

    output_file = '../../result_graphs/eps/noise_robot_h4samples_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

def GetRobotTotalRegrets_beta2():
    seeds = range(35)
    batch_size = 5
    """
    time_slot = 16
    root_path = '../../noise_robot_tests/beta2_fixed_exp/'
    root_path = '../../noise_robot_tests/beta2/'
    root_path = '../../noise_robot_tests/beta222/'
    # root_path = '../../noise_robot_tests/beta2_fixed/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    beta_list = [0.5, 1.0, 1.5, 2.0, 5.0]
    root_path = '../../noise_robot_tests/release/beta2_release-r/'
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot/unfixed_robot_beta2_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_beta3():
    seeds = range(35)
    batch_size = 5
    """
    time_slot = 16
    root_path = '../../noise_robot_tests/beta3_fixed_exp/'
    root_path = '../../noise_robot_tests/beta3/'
    root_path = '../../noise_robot_tests/beta33/'
    root_path = '../../noise_robot_tests/beta3_release/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]

    root_path = '../../noise_robot_tests/temp/'
    beta_list = [0.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../noise_robot_tests/release/beta3_release-r/'
    beta_list = [0.5, 1.0, 1.5, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot/unfixed_robot_beta3_simple_regrets.eps'
    # output_file = '../../result_graphs/eps/robot_beta3_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


if __name__ == "__main__":
    # GetRobotTotalRewards()
    # GetRobotTotalRegrets()
    # GetRobotBeta2Rewards()
    # GetRobotBeta3Rewards()
    GetRobot_H2Full_TotalRewards()
    GetRobotTotalRegrets_H2Full()
    # GetRobot_H4Samples_TotalRewards()
    # GetRobotBeta2Rewards()
