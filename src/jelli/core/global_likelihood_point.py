
class GlobalLikelihoodPoint:


    def __init__(self, global_likelihood_instance, par_array, scale):

        self.global_likelihood_instance = global_likelihood_instance
        self.par_array = par_array
        self.scale = scale
        self._log_likelihood_dict = None
        self._chi2_dict = None
        self._NP_covariance = None

    def log_likelihood_dict(self, NP_covariance=False):

        if self._log_likelihood_dict is None or self._NP_covariance != NP_covariance:
            self._log_likelihood_dict = self.global_likelihood_instance._delta_log_likelihood_dict(
                self.par_array, self.scale, NP_covariance
                )
            self._NP_covariance = NP_covariance
        return self._log_likelihood_dict

    def log_likelihood_global(self, NP_covariance=False):

        return self._log_likelihood_dict(NP_covariance)['global']

    def chi2_dict(self, NP_covariance=False):

        if self._chi2_dict is None or self._NP_covariance != NP_covariance:
            self._chi2_dict = self.global_likelihood_instance._chi2_dict(
                self.par_array, self.scale, NP_covariance
                )
            self._NP_covariance = NP_covariance
        return self._chi2_dict
