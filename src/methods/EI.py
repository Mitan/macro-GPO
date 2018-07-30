import math

from src.Utils import EI_Acquizition_Function


def method_EI(x_0, gp, next_states):
    best_observation = max(x_0.history.measurements)

    # next_states = self.GetNextAugmentedStates(x_0)

    vBest = - float('inf')
    xBest = None

    current_locations = x_0.history.locations
    current_chol = gp.Cholesky(x_0.history.locations)

    for x_next in next_states:

        next_physical_state = x_next.physical_state
        var = gp.GPVariance(locations=current_locations, current_location=next_physical_state,
                            cholesky=current_chol)
        sigma = math.sqrt(var[0, 0])

        weights = gp.GPWeights(locations=current_locations, current_location=next_physical_state,
                               cholesky=current_chol)
        mu = gp.GPMean(measurements=x_0.history.measurements, weights=weights)[0]

        expectedImprov = EI_Acquizition_Function(mu=mu, sigma=sigma, best_observation=best_observation)
        if expectedImprov >= vBest:
            vBest = expectedImprov
            xBest = x_next

    return vBest, xBest, -1
