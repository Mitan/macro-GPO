import numpy as np
import math
import random


# given a list of macroactions, find the set of unique points at step i
def __GetSetOfNextPoints(available_states, step):
    # available states should be a list of type AugmentedState
    next_points_tuples = map(tuple, [next_state.physical_state[step, :] for next_state in available_states])
    # a set of tuples
    next_points = set(next_points_tuples)
    return map(lambda x: np.atleast_2d(x), list(next_points))


def method_BUCB(x_0, gp, t, available_states, batch_size, domain_size):
    tolerance_eps = 10 ** (-8)

    # available_states = self.GetNextAugmentedStates(x_0)

    if not available_states:
        raise Exception("BUCB could not move from  location " + str(x_0.physical_state))

    # first_points = self.GetSetOfNextPoints(available_states, 0)

    domain_size = domain_size
    delta = 0.1
    beta_multiplier = 0.2

    history_locations = x_0.history.locations

    # get the mu values for the first point
    mu_values = {}
    current_chol = gp.Cholesky(x_0.history.locations)
    first_points = __GetSetOfNextPoints(available_states, 0)
    for first_point in first_points:
        weights = gp.GPWeights(locations=history_locations, current_location=first_point,
                               cholesky=current_chol)
        mu = gp.GPMean(measurements=x_0.history.measurements, weights=weights)
        mu_values[tuple(first_point[0])] = mu

    for num_steps in range(batch_size):
        value_dict = {}
        best_next_value = - float("inf")
        for next_state in available_states:
            current_locations = np.append(history_locations, next_state.physical_state[:num_steps, :], axis=0)

            current_chol = gp.Cholesky(current_locations)
            next_point = next_state.physical_state[num_steps: num_steps + 1, :]
            Sigma = gp.GPVariance(locations=current_locations, current_location=next_point,
                                  cholesky=current_chol)

            first_point = next_state.physical_state[0:1, :]
            mu = mu_values[tuple(first_point[0])]
            iteration = batch_size * t + num_steps + 1
            beta_t1 = 2 * beta_multiplier * math.log(domain_size * (iteration ** 2) * (math.pi ** 2) / (6 * delta))
            predicted_val = mu[0] + math.sqrt(beta_t1) * math.sqrt(Sigma[0, 0])

            best_next_value = max(predicted_val, best_next_value)

            value_dict[tuple(map(tuple, next_state.physical_state))] = predicted_val

        available_states = [next_state for next_state in available_states if
                            abs(value_dict[tuple(map(tuple, next_state.physical_state))]
                                - best_next_value) < tolerance_eps]

    return -1.0, random.choice(available_states), -1.0
