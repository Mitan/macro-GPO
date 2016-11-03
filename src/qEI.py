import numpy as np
from scipy.stats import mvn


def qEI(Sigma, eps, mu, plugin, q):
        # hardcoded lower limit for multinormal pdf integration
        #todo
        # note that here we explicitly assume data is 2-d
        low = np.array([-20.0 for i in range(q)])

        pk = np.zeros((q,))
        first_term = np.zeros((q,))
        second_term = np.zeros((q,))
        for k in range(q):
            Sigma_k = covZk(Sigma, k)

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
                if i!= index and j!= index:
                    result[i,j] = sigma[i,j] + sigma[index, index] - sigma[index, i] - sigma[index, j]
                    result[j,i] = sigma[i,j] + sigma[index, index] - sigma[index, i] - sigma[index, j]
        return result