import numpy as np
import math
import random


def method_BUCB_PE(x_0, gp, t, available_states, batch_size, domain_size):
    tolerance_eps = 10 ** (-8)

    # available_states = self.GetNextAugmentedStates(x_0)

    if not available_states:
        raise Exception("BUCB-PE could not move from  location " + str(x_0.physical_state))

    domain_size = domain_size
    delta = 0.1
    t_squared = (t + 1) ** 2
    beta_t1 = 2 * math.log(domain_size * t_squared * (math.pi ** 2) / (6 * delta))

    best_current_measurement = - float("inf")
    history_locations = x_0.history.locations
    current_chol = gp.Cholesky(x_0.history.locations)

    predict_val_dict = {}
    # first step is ucb
    for next_state in available_states:
        first_point = next_state.physical_state[0:1, :]
        Sigma = gp.GPVariance(locations=history_locations, current_location=first_point,
                              cholesky=current_chol)
        weights = gp.GPWeights(locations=history_locations, current_location=first_point,
                               cholesky=current_chol)
        mu = gp.GPMean(measurements=x_0.history.measurements, weights=weights)

        predicted_val = mu[0] + math.sqrt(beta_t1) * math.sqrt(Sigma[0, 0])

        predict_val_dict[tuple(first_point[0])] = predicted_val
        if predicted_val > best_current_measurement:
            best_current_measurement = predicted_val

    # the states with selected several points according to batch construction
    available_states = [next_state for next_state in available_states if
                        abs(predict_val_dict[tuple(next_state.physical_state[0, :])]
                            - best_current_measurement) < tolerance_eps]

    # Pure exploration part
    for num_steps in range(1, batch_size):
        sigma_dict = {}
        best_next_sigma = - float("inf")

        for next_state in available_states:
            current_locations = np.append(history_locations, next_state.physical_state[:num_steps, :], axis=0)

            current_chol = gp.Cholesky(current_locations)
            next_point = next_state.physical_state[num_steps: num_steps + 1, :]
            # current_points = self.GetSetOfNextPoints(available_states, num_steps)
            sigma = gp.GPVariance(locations=current_locations, current_location=next_point,
                                  cholesky=current_chol)[0, 0]
            best_next_sigma = max(sigma, best_next_sigma)
            sigma_dict[tuple(map(tuple, next_state.physical_state))] = sigma

        available_states = [next_state for next_state in available_states if
                            abs(sigma_dict[tuple(map(tuple, next_state.physical_state))]
                                - best_next_sigma) < tolerance_eps]

    return -1.0, random.choice(available_states), -1.0
