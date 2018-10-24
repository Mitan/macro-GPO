from src.enum.CovarianceEnum import CovarianceEnum
from src.gp.covariance.SquareExponential import SquareExponential


class CovarianceGenerator:

    def __init__(self, hyper_storer):
        self.hyper_storer = hyper_storer

    def get_covariance(self):
        covariance_type = self.hyper_storer.type
        if covariance_type == CovarianceEnum.SquareExponential:
            return self.__get_square_exponential_covariance(self.hyper_storer)
        else:
            raise ValueError("Unknown dataset")

    def __get_square_exponential_covariance(self, hyper_storer):
        return SquareExponential(length_scale=hyper_storer.length_scale, signal_variance=hyper_storer.signal_variance)
