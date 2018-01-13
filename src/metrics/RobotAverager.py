from GeneralResultsAverager import RobotRewards


def GetRobotBeta2Rewards():
    seeds = range(35)
    seeds = range(35)

    root_path = '../../releaseTests/robot/beta_new2/'
    root_path = '../../robot_tests/beta2/'
    # root_path = '../../robot_tests/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    batch_size = 5
    time_slot = 16

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot_beta2_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, isBeta = True)


def GetRobotBeta3Rewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([33, 34]))
    root_path = '../../releaseTests/robot/beta_new3/'
    root_path = '../../robot_tests/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0,  5.0]

    beta_list = [0.0, 0.05, 0.1]

    batch_size = 5
    time_slot = 16

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot_beta3_rewards.eps'
    output_file = '../../result_graphs/eps/new_robot_beta3_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, isBeta = True)


def GetRobot_H4Samples_TotalRewards():
    seeds = range(35)
    batch_size = 5

    time_slot = 16

    methods = ['new_anytime_h4_ 5', 'anytime_h4_ 50', 'anytime_h4']
    methods = ['new_anytime_h4_ 5', 'anytime_h4_5' ,'anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    method_names = [r'$N = 5$', r'$N = 5$a' r'$N = 300$']

    root_path = '../../robot_tests/h4_samples/'

    output_file = '../../result_graphs/eps/robot_h4samples_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)

def GetRobotTotalRewards():
    seeds = range(35)
    batch_size = 5

    time_slot = 16

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'r_qei',  'fixed_pe', 'gp-bucb']
    # methods = ['new_anytime_h1', 'new_anytime_h2', 'new_anytime_h3']

    method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$', r'$H^* = 4$', r'MLE $H = 4$', 'qEI', 'GP-BUCB-PE', 'GP-BUCB']
    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']
    # method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$']
    root_path = '../../releaseTests/robot/slot_16/'
    # root_path = '../../robot_tests/tests3/'

    output_file = '../../result_graphs/eps/robot_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)


def GetRobot_H2Full_TotalRewards():
    seeds = range(35)
    batch_size = 5
    time_slot = 16

    # methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4']
    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    method_names = [r'$H^* = 2$ (all MA)', r'$H^* = 2$ (selected MA)', r'$H^* = 4$ (selected MA)']
    method_names = [r'Anytime-$\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                    'EI (all MA)']

    root_path = '../../releaseTests/robot/h2_full/'
    # root_path = '../../robot_tests/21_full/'

    output_file = '../../result_graphs/eps/robot_h2_full_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)


if __name__ == "__main__":

    GetRobotTotalRewards()
    GetRobotBeta2Rewards()
    GetRobotBeta3Rewards()
    GetRobot_H2Full_TotalRewards()

    GetRobot_H4Samples_TotalRewards()
