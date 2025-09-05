import pandas as pd
from collections import OrderedDict, defaultdict
from math import ceil
import numpy as np

class GlobalLikelihoodPoint:


    def __init__(self, global_likelihood_instance, par_array, scale, par_dep_cov=False):

        self.global_likelihood_instance = global_likelihood_instance
        self.par_array = par_array
        self.scale = scale
        self.par_dep_cov = par_dep_cov

        (
            self.prediction_no_theory_uncertainty,
            self.prediction_correlated,
            self.log_likelihood_no_th_unc_univariate,
            self.log_likelihood_no_th_unc_multivariate,
            self.log_likelihood_correlated,
            self.log_likelihood_summed,
            self.std_sm_exp_correlated_scaled,
        ) = self.global_likelihood_instance._log_likelihood_point(
            self.par_array,
            self.scale,
            par_dep_cov=self.par_dep_cov
        )
        self._log_likelihood_dict = None
        self._chi2_dict = None
        self._obstable_tree_cache = None

    def log_likelihood_dict(self):

        if self._log_likelihood_dict is None:
            delta_log_likelihood = self.log_likelihood_summed - self.global_likelihood_instance.sm_log_likelihood_summed
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
            chi2 = -2*self.log_likelihood_summed
            self._chi2_dict = dict(
                zip(
                    self.global_likelihood_instance.likelihoods,
                    chi2
                )
            )
        return self._chi2_dict

    def _obstable_tree(self):
        if self._obstable_tree_cache is None:
            obstable_tree = tree()

            (
                log_likelihood_no_th_unc_multivariate,
                log_likelihood_no_th_unc_multivariate_no_corr,
                log_likelihood_correlated,
                log_likelihood_correlated_no_corr,
                exp_central_correlated,
                std_th_exp_correlated,
            ) = self.global_likelihood_instance._obstable(
                self.prediction_no_theory_uncertainty,
                self.prediction_correlated,
                self.log_likelihood_no_th_unc_multivariate,
                self.log_likelihood_correlated,
                self.std_sm_exp_correlated_scaled,
            )

            pull_sm_no_theory_uncertainty_no_corr, pull_exp_no_theory_uncertainty_no_corr = compute_pulls(
                self.log_likelihood_no_th_unc_univariate + log_likelihood_no_th_unc_multivariate_no_corr,
                self.global_likelihood_instance.sm_log_likelihood_no_theory_uncertainty_no_corr
            )

            pull_sm_no_theory_uncertainty, pull_exp_no_theory_uncertainty = compute_pulls(
                self.log_likelihood_no_th_unc_univariate + log_likelihood_no_th_unc_multivariate,
                self.global_likelihood_instance.sm_log_likelihood_no_theory_uncertainty
            )

            # add no theory uncertainty observables
            experimental_values_no_theory_uncertainty = self.global_likelihood_instance.experimental_values_no_theory_uncertainty
            for i, obs_name in enumerate(self.global_likelihood_instance.observables_no_theory_uncertainty):
                obstable_tree[obs_name] = {
                    "name": obs_name,
                    "experiment": experimental_values_no_theory_uncertainty[obs_name][0],
                    "exp. unc.": experimental_values_no_theory_uncertainty[obs_name][1],
                    "theory": self.prediction_no_theory_uncertainty[i],
                    "th. unc.": 0.0,
                    "pull exp.": pull_exp_no_theory_uncertainty_no_corr[i],
                    "pull SM": pull_sm_no_theory_uncertainty_no_corr[i],
                    "pull exp. corr": pull_exp_no_theory_uncertainty[i],
                    "pull SM corr": pull_sm_no_theory_uncertainty[i],
                }

            # add correlated observables
            for n_obs_sector, obs_names in enumerate(self.global_likelihood_instance.observables_correlated):
                prediction_correlated = self.prediction_correlated[n_obs_sector][0]
                pull_sm_correlated_no_corr, pull_exp_correlated_no_corr = compute_pulls(
                    log_likelihood_correlated_no_corr[n_obs_sector],
                    self.global_likelihood_instance.sm_log_likelihood_correlated_no_corr[n_obs_sector]
                )
                pull_sm_correlated, pull_exp_correlated = compute_pulls(
                    log_likelihood_correlated[n_obs_sector],
                    self.global_likelihood_instance.sm_log_likelihood_correlated[n_obs_sector]
                )
                experiment = exp_central_correlated[n_obs_sector]
                std_th_exp = std_th_exp_correlated[n_obs_sector]
                std_exp = self.global_likelihood_instance.std_exp[n_obs_sector]
                std_th = std_th_exp*np.sqrt(1 - (std_exp/std_th_exp)**2)
                for i, obs_name in enumerate(obs_names):
                    obstable_tree[obs_name] = {
                        "name": obs_name,
                        "experiment": experiment[i],
                        "exp. unc.": std_exp[i],
                        "theory": prediction_correlated[i],
                        "th. unc.": std_th[i],
                        "pull exp.": pull_exp_correlated_no_corr[i],
                        "pull SM": pull_sm_correlated_no_corr[i],
                        "pull exp. corr": pull_exp_correlated[i],
                        "pull SM corr": pull_sm_correlated[i],
                    }

            self._obstable_tree_cache = obstable_tree
        return self._obstable_tree_cache

    # TODO: this is mostly copy paste from smelli, we could think if something should be changed
    def obstable(self, min_pull_exp=0, sort_by='pull exp.', ascending=None, min_val=None, max_val=None):

        sort_keys = ['name', 'exp. unc.', 'experiment', 'pull SM', 'pull exp.', 'th. unc.', 'theory', 'pull exp. corr', 'pull SM corr']
        if sort_by not in sort_keys:
            raise ValueError(
                "'{}' is not an allowed value for sort_by. Allowed values are "
                "'{}', and '{}'.".format(sort_by, "', '".join(sort_keys[:-1]),
                                        sort_keys[-1])
            )
        subset = None
        if sort_by == 'pull exp.':
            # if sorted by pull exp., use descending order as default
            if ascending is None:
                ascending = False
            if min_val is not None:
                min_val = max(min_pull_exp, min_val)
            else:
                min_val = min_pull_exp
        elif min_pull_exp != 0:
            subset = lambda row: row['pull exp.'] >= min_pull_exp
        # if sorted not by pull exp., use ascending order as default
        if ascending is None:
            ascending = True
        obstable_tree = self._obstable_filter_sort(
            self._obstable_tree(),
            sortkey=sort_by,
            ascending=ascending,
            min_val=min_val,
            max_val=max_val,
            subset=subset
        )
        df = pd.DataFrame(obstable_tree).T
        if len(df) >0:
            del(df['name'])
        return df

    @staticmethod
    def _obstable_filter_sort(info, sortkey='name', ascending=True, min_val=None, max_val=None, subset=None, max_rows=None):
        # impose min_val and max_val
        if min_val is not None:
            info = {obs:row for obs,row in info.items()
                    if row[sortkey] >= min_val}
        if max_val is not None:
            info = {obs:row for obs,row in info.items()
                    if row[sortkey] <= max_val}
        # get only subset:
        if subset is not None:
            info = {obs:row for obs,row in info.items() if subset(row)}
        # sort
        info = OrderedDict(sorted(info.items(), key=lambda x: x[1][sortkey],
                                reverse=(not ascending)))
        # restrict number of rows per tabular to max_rows
        if max_rows is None or len(info)<=max_rows:
            return info
        else:
            info_list = []
            for n in range(ceil(len(info)/max_rows)):
                info_n = OrderedDict((obs,row)
                                     for i,(obs,row) in enumerate(info.items())
                                     if i>=n*max_rows and i<(n+1)*max_rows)
                info_list.append(info_n)
            return info_list

def tree():
    return defaultdict(tree)

def compute_pulls(log_likelihood, log_likelihood_sm):
    s = np.where(log_likelihood > log_likelihood_sm, -1, 1)
    pull_sm = s * np.sqrt(np.abs(-2 * (log_likelihood - log_likelihood_sm)))
    pull_exp = np.sqrt(np.abs(-2 * log_likelihood))
    return pull_sm, pull_exp
