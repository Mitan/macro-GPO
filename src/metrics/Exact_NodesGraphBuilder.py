from src.PlottingEnum import PlottingMethods
from src.ResultsPlotter import PlotData


def get_nodes_number_array(H, actions, samples):
    # actions -  number of macro-actions
    # samples - number of samples at every node
    h1 = actions
    h2 = actions * samples + h1 * samples * actions
    h3 = actions * samples + h2 * samples * actions
    h4 = actions * samples + h3 * samples * actions
    if H == 4:
        return [h4, h4, h3, h2, h1]
    elif H == 3:
        return [h3, h3, h3, h2, h1]
    elif H == 2:
        return [h2, h2, h2, h2, h1]
    elif H == 1:
        return [h1, h1, h1, h1, h1]


results = []
"""
for i in range(4):
    h_current = get_nodes_number_array(H=i + 1, actions=4, samples=100)
    results.append(['H = ' + str(i + 1), h_current])
    print sum(h_current)

print results
"""
N_samples = [5, 20, 100]
for N in N_samples:
    h_current = get_nodes_number_array(H=4, actions=4, samples=N)
    results.append([r'$N$ = ' + str(N), h_current])
    print sum(h_current)

output_filename = '../../result_graphs/eps/simulated_nodes.eps'

PlotData(results=results, output_file_name=output_filename, plottingType=PlottingMethods.Nodes, dataset='simulated')

