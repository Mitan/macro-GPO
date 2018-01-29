import GPy
import GPyOpt
from GPyOpt.models import GPModel
from numpy.random import seed
import numpy as np

# by default searching for max.
# if need min, set maximize=True
# then it will maximize and output -f
# still find max of it and invert

objective_true = GPyOpt.objective_examples.experiments2d.branin()  # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd=0.1)  # noisy version
bounds = objective_noisy.bounds

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]},  ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]



# objective_true.plot()
batch_size = 5
num_cores = 4
"""
locations = np.array([[3, 12], [4, 9], [0, 0]])
locations = np.array([[3, 12], [3, 11]])
Y = objective_noisy.f(locations)

num_points = 100
s1 = np.random.randint(0, 1000, size=num_points)
s2 = np.random.randint(0, 1000, size=num_points)


x = zip(s1, s2)

# print x
dt=np.dtype('float,float')
x = np.array(zip(s1, s2), dtype=dt)

locations = x[:4]
print locations
# print locations
# domain = [{'name': 'stations', 'type': 'bandit', 'domain':x }]

"""
n_points = 10
domain = [{'name': 'var_1', 'type': 'discrete', 'domain': range(0,n_points)},  ## use default bounds
          {'name': 'var_2', 'type': 'discrete', 'domain': range(0, n_points)}]

y1 = np.random.random(size=n_points**2)

dic = {}
for i in range(n_points):
    for j in range(n_points):
        dic[(i,j)] = y1[i]

for k in dic:
    pass
    # print k, dic[k]


def my_f(x):
    # print x[0]
    return dic[tuple(x[0])]
"""
locations = np.array([[3, 0], [4, 7], [0, 0]])
Y = np.array([[0.5], [0.5], [0,5]])
print Y.shape
"""
locations = np.array([[3, 12], [4, 9]])

# shape needs to be (num_init_points, 1)
Y = objective_noisy.f(locations)


n_points = 10
domain = [{'name': 'var_1', 'type': 'discrete', 'domain': range(0,n_points)},  ## use default bounds
          {'name': 'var_2', 'type': 'discrete', 'domain': range(0, n_points)}]


seed(123)

signal = 1.0
noise = 0.1
l = [1.1, 2.1]
kernel = GPy.kern.RBF(input_dim=2, variance=signal, lengthscale=l, ARD=True)
m = GPModel(kernel=kernel, noise_var=noise)

# BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,


iter_count = 3
current_iter = 0
X_step = np.array([[3, 12], [4, 9]])
X_step = np.array([3, 12])
Y_step = objective_noisy.f(locations)
ignored_X = np.array([(0.0, 2.0)], dtype=np.dtype('float,float'))
ignored_X = None


while current_iter < iter_count:
    # bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X_step, Y=Y_step)
    BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=None,
                                                           domain=domain,
                                                           acquisition_type='EI',
                                                           normalize_Y=True,
                                                           initial_design_numdata=4,
                                                           evaluator_type='local_penalization',
                                                           batch_size=batch_size,
                                                           num_cores=num_cores,
                                                           acquisition_jitter=0,
                                                           X=X_step, Y=Y_step,
                                                           kernel=kernel,
                                                           maximize=False,
                                                           de_duplication=True)
    x_next = BO_demo_parallel.suggest_next_locations(ignored_X = ignored_X)
    y_next = objective_noisy.f(x_next)

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

    current_iter += 1

"""
max_iter = 1
BO_demo_parallel.run_optimization(max_iter)


# bo = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step, de_duplication = True)
BO_demo_parallel.suggest_next_locations(ignored_X = ignored_X)
"""
print BO_demo_parallel.model.model
print BO_demo_parallel.get_evaluations()
# BO_demo_parallel._print_convergence()
# BO_demo_parallel.plot_acquisition()
