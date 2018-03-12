import numpy as np
from scipy.stats import mvn


def method_qEI(x_0, next_states, gp, batch_size, eps=10 ** (-5)):
    # x_0 stores a 2D np array of k points with history
    max_measurement = max(x_0.history.measurements)

    best_action = None
    best_expected_improv = - float("inf")
    # best_expected_improv = -1.0

    # valid_actions = self.GetValidActionSet(x_0.physical_state)
    # next_states = [self.TransitionP(x_0, a) for a in valid_actions]

    # next_states = self.GetNextAugmentedStates(x_0)
    if not next_states:
        raise Exception("qEI could not move from   location " + str(x_0.physical_state))

    chol = gp.Cholesky(x_0.history.locations)

    for x_next in next_states:
        # x_next = self.TransitionP(x_0, a)

        Sigma = gp.GPVariance(locations=x_next.history.locations,
                              current_location=x_next.physical_state,
                              cholesky=chol)
        weights = gp.GPWeights(locations=x_next.history.locations, current_location=x_next.physical_state,
                               cholesky=chol)
        mu = gp.GPMean(measurements=x_next.history.measurements, weights=weights)

        expectedImprov = qEI(Sigma, eps, mu, max_measurement, batch_size)

        # comparison
        if expectedImprov >= best_expected_improv:
            best_expected_improv = expectedImprov
            best_action = x_next

    return best_expected_improv, best_action, len(next_states)


def qEI(sigma, eps, mu, plugin, q):
    # hardcoded lower limit for multinormal pdf integration
    # todo
    # note that here we explicitly assume data is 2-d
    low = np.array([-20.0 for _ in range(q)])

    pk = np.zeros((q,))
    first_term = np.zeros((q,))
    second_term = np.zeros((q,))
    for k in range(q):
        Sigma_k = covZk(sigma, k)

        mu_k = mu - mu[k]
        mu_k[k] = - mu[k]
        b_k = np.zeros((q,))
        b_k[k] = - plugin

        # suspect that it is equal to pmnorm function of original R package
        zeros = np.array([0.0 for i in range(q)])
        p, _ = mvn.mvnun(low, b_k - mu_k, zeros, Sigma_k)
        pk[k] = p
        first_term[k] = (mu[k] - plugin) * pk[k]

        upper_temp = b_k + eps * Sigma_k[:, k] - mu_k
        p1, _ = mvn.mvnun(low, upper_temp, zeros, Sigma_k)
        second_term[k] = 1 / eps * (p1 - pk[k])
    expectedImprov = np.sum(first_term) + np.sum(second_term)
    return expectedImprov


def covZk(sigma, index):
    q = sigma.shape[0]
    result = np.zeros(sigma.shape)
    result[index, index] = sigma[index, index]
    for i in range(q):
        if i != index:
            result[index, i] = sigma[index, index]
            result[i, index] = sigma[index, index]
        for j in range(i, q):
            if i != index and j != index:
                result[i, j] = sigma[i, j] + sigma[index, index] - sigma[index, i] - sigma[index, j]
                result[j, i] = sigma[i, j] + sigma[index, index] - sigma[index, i] - sigma[index, j]
    return result
