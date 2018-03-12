import math
import random

import numpy as np
from scipy.stats import norm
from scipy import special


def __get_gp_prediction(current_point, history_locations, history_measurements, cholesky, gp):
    Sigma = gp.GPVariance(locations=history_locations, current_location=current_point,
                          cholesky=cholesky)[0, 0]
    weights = gp.GPWeights(locations=history_locations, current_location=current_point,
                           cholesky=cholesky)
    mu = gp.GPMean(measurements=history_measurements, weights=weights)[0]
    return mu, Sigma


# acquizition function
def __transformed_ei(current_point, history_locations, history_measurements, max_measurement, cholesky, gp):
    # first_point = next_state.physical_state[0:1, :]
    mu, Sigma = __get_gp_prediction(current_point=current_point,
                                    history_locations=history_locations,
                                    history_measurements=history_measurements,
                                    cholesky=cholesky,
                                    gp=gp)
    # predicted_val = mu[0] + math.sqrt(beta_t1) * math.sqrt(Sigma[0, 0])
    Z = (mu - max_measurement) / Sigma
    predicted_val = (mu - max_measurement) * norm.cdf(x=Z, loc=0, scale=1.0) \
                    + Sigma * norm.pdf(x=Z, loc=0, scale=1.0)

    # print predicted_val, mu, Sigma
    # transform ei value
    if predicted_val <= 0:
        predicted_val = math.log(1 + math.exp(predicted_val))
    return predicted_val


# mu and var a posterior mean and variance in old_x
def __local_penalizer(new_x, old_x, mu, var, M):
    # todo note hardcoded
    L = 10

    z = 1 / math.sqrt(2 * var) * (L * np.linalg.norm(new_x - old_x) - M + mu)
    return 0.5 * special.erfc(-z)


def method_LP(x_0, available_states, gp, batch_size):
    M = max(x_0.history.measurements)
    L = 10

    tolerance_eps = 10 ** (-8)

    # available_states = self.GetNextAugmentedStates(x_0)

    if not available_states:
        raise Exception("LP could not move from  location " + str(x_0.physical_state))

    best_current_measurement = - float("inf")
    history_locations = x_0.history.locations
    history_measurements = x_0.history.measurements
    current_chol = gp.Cholesky(x_0.history.locations)

    predict_val_dict = {}
    # first step is ei
    for next_state in available_states:
        first_point = next_state.physical_state[0:1, :]
        predicted_val = __transformed_ei(current_point=first_point,
                                         history_locations=history_locations,
                                         history_measurements=history_measurements,
                                         max_measurement=M,
                                         cholesky=current_chol,
                                         gp=gp)
        print first_point[0]

        predict_val_dict[tuple(first_point[0])] = predicted_val
        if predicted_val > best_current_measurement:
            best_current_measurement = predicted_val

    # the states with selected several points according to batch construction
    available_states = [next_state for next_state in available_states if
                        abs(predict_val_dict[tuple(next_state.physical_state[0, :])]
                            - best_current_measurement) < tolerance_eps]

    # Pure exploration part
    for num_steps in range(1, batch_size):
        penalized_value_dict = {}
        best_next_value = - float("inf")

        for next_state in available_states:
            # current_locations = np.append(history_locations, next_state.physical_state[:num_steps, :], axis=0)

            # current_chol = self.gp.Cholesky(current_locations)
            next_point = next_state.physical_state[num_steps: num_steps + 1, :]
            ac_value = __transformed_ei(current_point=next_point,
                                        history_locations=history_locations,
                                        history_measurements=history_measurements,
                                        max_measurement=M,
                                        cholesky=current_chol,
                                        gp=gp)
            print ac_value
            # old points
            for j in range(num_steps):
                x_j = next_state.physical_state[j: j + 1, :]
                mu_j, Sigma_j = __get_gp_prediction(current_point=x_j,
                                                    history_locations=history_locations,
                                                    history_measurements=history_measurements,
                                                    cholesky=current_chol,
                                                    gp=gp)
                penalizer_j = __local_penalizer(new_x=next_point, old_x=x_j, mu=mu_j,
                                                var=Sigma_j, M=M)

                ac_value = ac_value * penalizer_j

            best_next_value = max(ac_value, best_next_value)
            penalized_value_dict[tuple(map(tuple, next_state.physical_state))] = ac_value

        available_states = [next_state for next_state in available_states if
                            abs(penalized_value_dict[tuple(map(tuple, next_state.physical_state))]
                                - best_next_value) < tolerance_eps]

    return -1.0, random.choice(available_states), -1.0
