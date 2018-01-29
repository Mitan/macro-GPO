import GPy
import GPyOpt
import numpy as np


objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd=0.1)  # noisy version
objective_true = GPyOpt.objective_examples.experiments2d.branin()

batch_size = 5
num_cores = 4
n_points = 10
domain = [{'name': 'var_1', 'type': 'discrete', 'domain': range(10)},  ## use default bounds
          {'name': 'var_2', 'type': 'discrete', 'domain': range(10)}]


current_iter = 0
X_step = np.array([[3, 12], [4, 9]])
Y_step = objective_noisy.f(X_step)
ignored_X = np.array([[0.0, 2.0]])



while current_iter < 3:
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
                                                           maximize=False,
                                                           de_duplication=True)
    x_next = BO_demo_parallel.suggest_next_locations(ignored_X = ignored_X)
    y_next = objective_noisy.f(x_next)

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

    current_iter += 1

print BO_demo_parallel.get_evaluations()
