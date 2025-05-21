from typing import List, Dict, Tuple, Any, Callable, Union, Optional


class GlobalLikelihoodPoint:


    def __init__(self, global_likelihood_instance, par_array, scale):

        self.global_likelihood_instance = global_likelihood_instance
        self.par_array = par_array
        self.scale = scale
        self._log_likelihood_dict = None
        self._chi2_dict = None

    def log_likelihood_dict(self):

        if self._log_likelihood_dict is None:
            self._log_likelihood_dict = self.global_likelihood_instance._delta_log_likelihood_dict(
                self.par_array, self.scale
                )
        return self._log_likelihood_dict

    def log_likelihood_global(self):

        return self._log_likelihood_dict()['global']

    def chi2_dict(self):

        if self._chi2_dict is None:
            self._chi2_dict = self.global_likelihood_instance._chi2_dict(
                self.par_array, self.scale
                )
        return self._chi2_dict
