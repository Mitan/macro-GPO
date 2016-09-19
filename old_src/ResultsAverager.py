from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO

__author__ = 'Dmitrii'


basepath = './tests_2/function'


test_cases = [0,2,5]

funcs = ['_Branin', '_SixCamel',  '__Cosines', 'LogK']
funcs = ['_Branin', '_SixCamel',  '__Cosines', 'LogK']
funcs = ['_Branin']
#funcs = ['_SixCamel']
funcs = [ '__Cosines']
funcs = ['LogK']
beta_values = [0.0, 0.5, 1.0,  3.0]
locations = range(7)
methods = ['ei', 'h2_non-myopic', 'h3_non-myopic','ucb', 'h3_mle']
methods_names = ['qEI', 'H = 2', 'H = 3', 'UCB', 'H = 3 MLE']
#methods = ['h3_non-myopic']
"""
basepath = './testsBetaCosines/batch2/__Cosines/'
for beta in beta_values:
#for j in range(len(funcs)):
    #func_path = basepath  + funcs[j] + '/beta0.0/'
        results = []
        #for k in range(len(beta_values)):

    # for every method iterate over locations
    #for i in range(len(methods)):
        results_for_method = np.zeros((20,))
        number_of_location = 0
        for loc in range(10):


                # cheat cosines
                if loc == 12 or loc == 10:
                    continue
                # cheat SixCamel

                if loc == 9 or loc == 11:
                    continue

                if loc in [1,10,6, 7,15]:
                    continue

                #folder_path =  func_path +'location' + str(loc) + '/'
                folder_path =  basepath  +'/location' + str(loc) + '/beta' + str(beta) + '/'
                file_path = folder_path + methods[2] + '/summary.txt'
                try:
                        a = (open(file_path).readlines()[2])[1: -2]
                except:
                        continue
                print file_path
                number_of_location+=1
                a = StringIO(a)
                rewards = np.genfromtxt(a, delimiter=",")
                #result = np.genfromtxt(a)
                results_for_method = np.add(results_for_method, rewards)
    print number_of_location
    results_for_method = results_for_method / number_of_location
        #result = [methods_names[i], results_for_method.tolist()]
    result = ['beta = ' + str(beta)], results_for_method.tolist()]
    results.append(result)

    PlotData(results, basepath)
"""
basepath = './testsBetaNew/batch2/functionLogK/'
results = []
for beta in beta_values:
        results_for_method = np.zeros((15,))
        number_of_location = 0
        for loc in range(20):
                folder_path =  basepath  +'/location' + str(loc) + '/beta' + str(beta) + '/'
                # h3
                file_path = folder_path + methods[2] + '/summary.txt'
                try:
                        a = (open(file_path).readlines()[2])[1: -2]
                except:
                        continue
                print file_path
                number_of_location+=1
                a = StringIO(a)
                rewards = np.genfromtxt(a, delimiter=",")
                #result = np.genfromtxt(a)
                results_for_method = np.add(results_for_method, rewards)
        print number_of_location
        results_for_method = results_for_method / number_of_location
        #result = [methods_names[i], results_for_method.tolist()]
        result = ['beta = ' + str(beta), results_for_method.tolist()]
        results.append(result)

PlotData(results, basepath)
