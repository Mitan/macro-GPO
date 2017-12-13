import os

import numpy as np
import rpy2.robjects as robjects  # This initializes R
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


class newQEI:
    def __init__(self, length_scale, signal_variance, noise_variance, locations, Y):
        importr('DiceOptim')  # Import DiceOptim R package
        # Enables the conversion of numpy objects to rpy2 objects
        numpy2ri.activate()

        # Include the file r_callers to the R path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        robjects.r("source('" + dir_path + "/r_callers.R')")
        # Pack the parameters of the model in the format
        # required by DiceKriging
        """
        if self.kern.ARD:
            cov_param = np.asarray(self.kern.lengthscales.value)
        else:
        """
        cov_param = np.asarray(length_scale)
        cov_var = np.array((signal_variance,))
        var = np.array((noise_variance,))

        cov_type = 'gauss'  # DiceKriging calls the RBF kernel 'gauss'

        # Call R. Warning: the code in qEI_caller calculates the noiseless
        # expected improvement

        self.r_model = robjects.r['create_model'](
            locations, Y,
            cov_type, cov_param, cov_var, var
        )

    def acquisition(self, X):
        qEI = robjects.r['qEI_caller'](X, self.r_model)
        opt_val = np.array(qEI)

        return opt_val
