from src.metric.DatasetMaxExtractor import DatasetMaxExtractor
from src.metric.RegretCalculator import RegretCalculator
from src.enum.PlottingEnum import PlottingMethods
from src.enum.SinglePointMethodsDict import single_point_methods
from src.plotting.ResultsPlotter import PlotData


#### Simulated ####
def GetSimulatedTotalRegrets_H4Samples():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../simulated_tests/h4_samples/'
    root_path = '../../releaseTests/simulated/h4_samples/'
    
    methods = ['h4', 'new_new_h4_20', 'h4_5']
    method_names = ['N=100', 'N=20', 'N=5']
    output_file = '../../result_graphs/eps/simulated_h4_samples_simple_regrets.eps'

    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedBeta2Regrets():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    root_path = '../../simulated_tests/beta2/'

    beta_list = [ 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    
    root_path = '../../simulated_tests/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../simulated_tests/beta2-good/'

    beta_list = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' +  str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta2_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedBeta3Regrets():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]

    root_path = '../../simulated_tests/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    
    root_path = '../../simulated_tests/beta3_good_zero_mean/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../simulated_tests/beta3-good/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta3_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

#### Robot

if __name__ == "__main__":
    """
    # GetRoadTotalRegrets()
    # GetRoadTotalRegrets_H2Full()
    GetRoadBeta2Regrets()
    GetRoadBeta3Regrets()
    # GetSimulatedBeta2Regrets()
    # GetSimulatedBeta3Regrets()
    # GetRobotTotalRegrets_beta2()
    # GetRobotTotalRegrets_beta3()
   
    GetSimulatedTotalRegrets()
    GetSimulatedTotalRegrets_H4Samples()
    """
    """
    GetRobotTotalRegrets()
    GetRobotTotalRegrets_H2Full()
    GetRobotTotalRegrets_H4Samples()
    """

    """
    GetRoadTotalRegrets_H2Full_H4Samples()
    GetRobotTotalRegrets_H4Samples()
    """
    # GetRobotTotalRegrets()
    # GetRobotTotalRegrets_H2Full()