
class GlobalLikelihoodPoint:


    def __init__(self, global_likelihood_instance, par_array, scale, par_dep_cov=False):

        self.global_likelihood_instance = global_likelihood_instance
        self.par_array = par_array
        self.scale = scale
        self.par_dep_cov = par_dep_cov

        (
            self.prediction_no_theory_uncertainty,
            self.prediction_correlated,
            self.log_likelihood_univariate_per_observable,
            self.log_likelihood_multivariate_per_observable,
            self.log_likelihood_correlated_per_observable,
            self.log_likelihood,
            self.standard_deviation_th_correlated,
        ) = self.global_likelihood_instance._log_likelihood_point(
            self.par_array,
            self.scale,
            par_dep_cov=self.par_dep_cov
        )
        self._log_likelihood_dict = None
        self._chi2_dict = None

    def log_likelihood_dict(self):

        if self._log_likelihood_dict is None:
            delta_log_likelihood = self.log_likelihood - self.global_likelihood_instance.sm_log_likelihood
            self._log_likelihood_dict = dict(
                zip(
                    self.global_likelihood_instance.likelihoods,
                    delta_log_likelihood
                )
            )
        return self._log_likelihood_dict

    def log_likelihood_global(self):

        return self.log_likelihood_dict['global']

    def chi2_dict(self):
        if self._chi2_dict is None:
            chi2 = -2*self.log_likelihood
            self._chi2_dict = dict(
                zip(
                    self.global_likelihood_instance.likelihoods,
                    chi2
                )
            )
        return self._chi2_dict
