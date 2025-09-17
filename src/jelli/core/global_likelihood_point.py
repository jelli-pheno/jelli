import pandas as pd
from collections import OrderedDict, defaultdict
from math import ceil
import numpy as np

class GlobalLikelihoodPoint:
    '''
    A class to represent a point in the parameter space of the global likelihood.

    Parameters
    ----------
    global_likelihood_instance : GlobalLikelihood
        An instance of the `GlobalLikelihood` class.
    par_array : np.ndarray
        An array of parameter values.
    scale : float
        The scale at which the parameters are evaluated.
    par_dep_cov : bool, optional
        If `True`, use parameter-dependent covariance matrices. Default is `False`.

    Attributes
    ----------
    global_likelihood_instance : GlobalLikelihood
        The instance of the `GlobalLikelihood` class.
    par_array : np.ndarray
        The array of parameter values.
    scale : float
        The scale at which the parameters are evaluated.
    par_dep_cov : bool
        If `True`, use parameter-dependent covariance matrices.
    prediction_no_theory_uncertainty : np.ndarray
        The predictions for observables without theory uncertainty.
    prediction_correlated : List[np.ndarray]
        The predictions for observables with correlated uncertainties.
    log_likelihood_no_th_unc_univariate : np.ndarray
        The log-likelihood contributions from observables without theory uncertainty with univariate constraints.
    log_likelihood_no_th_unc_multivariate : List[np.ndarray]
        The log-likelihood contributions from observables without theory uncertainty with multivariate constraints.
    log_likelihood_correlated : List[np.ndarray]
        The log-likelihood contributions from observables with correlated theoretical uncertainties.
    log_likelihood_summed : np.ndarray
        The total log-likelihood summed over all observables.
    std_sm_exp_correlated_scaled : List[np.ndarray]
        The scaled total standard deviations for correlated observables.
    _log_likelihood_dict : dict
        A dictionary mapping likelihood names to their log-likelihood values.
    _chi2_dict : dict
        A dictionary mapping likelihood names to their chi-squared values.
    _obstable_tree_cache : defaultdict
        A cached tree structure for the observable table.

    Methods
    -------
    log_likelihood_dict()
        Returns a dictionary mapping likelihood names to their log-likelihood values.
    log_likelihood_global()
        Returns the global log-likelihood value.
    chi2_dict()
        Returns a dictionary mapping likelihood names to their chi-squared values.
    obstable(min_pull_exp=0, sort_by='pull exp.', ascending=None, min_val=None, max_val=None)
        Returns a pandas DataFrame representing the observable table with various filtering and sorting options.
    _obstable_tree()
        Constructs and returns a tree structure for the observable table.
    _obstable_filter_sort(info, sortkey='name', ascending=True, min_val=None, max_val=None, subset=None, max_rows=None)
        Filters and sorts the observable table based on specified criteria.

    Examples
    --------
    Initialize a `GlobalLikelihoodPoint` instance:

    >>> gl = GlobalLikelihood(...)
    >>> glp = GlobalLikelihoodPoint(gl, par_array=np.array([0.1, 0.2]), scale=1000)

    Access the log-likelihood dictionary:

    >>> log_likelihood_dict = glp.log_likelihood_dict()

    Access the global log-likelihood value:

    >>> log_likelihood_global = glp.log_likelihood_global()

    Access the chi-squared dictionary:

    >>> chi2_dict = glp.chi2_dict()

    Get the observable table:

    >>> obstable_df = glp.obstable()
    '''

    def __init__(self, global_likelihood_instance, par_array, scale, par_dep_cov=False):
        '''
        Initialize the `GlobalLikelihoodPoint` class.

        Parameters
        ----------
        global_likelihood_instance : GlobalLikelihood
            An instance of the `GlobalLikelihood` class.
        par_array : np.ndarray
            An array of parameter values.
        scale : float
            The scale at which the parameters are evaluated.
        par_dep_cov : bool, optional
            If `True`, use parameter-dependent covariance matrices. Default is `False`.

        Attributes
        ----------
        global_likelihood_instance : GlobalLikelihood
            The instance of the `GlobalLikelihood` class.
        par_array : np.ndarray
            The array of parameter values.
        scale : float
            The scale at which the parameters are evaluated.
        par_dep_cov : bool
            If `True`, use parameter-dependent covariance matrices.
        prediction_no_theory_uncertainty : np.ndarray
            The predictions for observables without theory uncertainty.
        prediction_correlated : List[np.ndarray]
            The predictions for observables with correlated uncertainties.
        log_likelihood_no_th_unc_univariate : np.ndarray
            The log-likelihood contributions from observables without theory uncertainty with univariate constraints.
        log_likelihood_no_th_unc_multivariate : List[np.ndarray]
            The log-likelihood contributions from observables without theory uncertainty with multivariate constraints.
        log_likelihood_correlated : List[np.ndarray]
            The log-likelihood contributions from observables with correlated theoretical uncertainties.
        log_likelihood_summed : np.ndarray
            The total log-likelihood summed over all observables.
        std_sm_exp_correlated_scaled : List[np.ndarray]
            The scaled total standard deviations for correlated observables.
        _log_likelihood_dict : dict
            A dictionary mapping likelihood names to their log-likelihood contributions.
        _chi2_dict : dict
            A dictionary mapping likelihood names to their chi-squared contributions.
        _obstable_tree_cache : defaultdict
            A cached tree structure for the observable table.

        Examples
        --------
        Initialize a `GlobalLikelihoodPoint` instance:

        >>> gl = GlobalLikelihood(...)
        >>> glp = GlobalLikelihoodPoint(gl, par_array=np.array([0.1, 0.2]), scale=1000)
        '''
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
        '''
        Returns a dictionary mapping likelihood names to their log-likelihood values.

        Returns
        -------
        dict
            A dictionary where keys are likelihood names and values are their log-likelihood values.

        Examples
        --------
        Access the log-likelihood dictionary:

        >>> log_likelihood_dict = glp.log_likelihood_dict()
        '''
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
        '''
        Returns the global log-likelihood value.

        Returns
        -------
        float
            The global log-likelihood value.

        Examples
        --------
        Access the global log-likelihood value:

        >>> log_likelihood_global = glp.log_likelihood_global()
        '''
        return self.log_likelihood_dict['global']

    def chi2_dict(self):
        '''
        Returns a dictionary mapping likelihood names to their chi-squared values.

        Returns
        -------
        dict
            A dictionary where keys are likelihood names and values are their chi-squared values.

        Examples
        --------
        Access the chi-squared dictionary:

        >>> chi2_dict = glp.chi2_dict()
        '''
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
        '''
        Constructs and returns a tree structure for the observable table.

        Returns
        -------
        defaultdict
            A tree structure containing observable names, constraints, predictions, uncertainties, and pulls.
        '''
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
        '''
        Returns a pandas DataFrame representing the observable table with various filtering and sorting options. The table includes observable names, experimental values, uncertainties, theoretical predictions, and pulls.

        Parameters
        ----------
        min_pull_exp : float, optional
            Minimum absolute value of the experimental pull to include an observable. Default is `0`.
        sort_by : str, optional
            The column by which to sort the DataFrame. Options are `'name'`, `'exp. unc.'`, `'experiment'`, `'pull SM'`, `'pull exp.'`, `'th. unc.'`, `'theory'`, `'pull exp. corr'`, and `'pull SM corr'`. Default is `'pull exp.'`.
        ascending : bool, optional
            If `True`, sort in ascending order. If `False`, sort in descending order. If `None`, the default sorting order is used based on the `sort_by` parameter. Default is `None`.
        min_val : float, optional
            Minimum value for the `sort_by` column to include an observable. Default is `None`.
        max_val : float, optional
            Maximum value for the `sort_by` column to include an observable. Default is `None`.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame representing the observable table with the specified filtering and sorting applied.

        Examples
        --------
        Get the observable table sorted by experimental pull in descending order:

        >>> obstable_df = glp.obstable(sort_by='pull exp.', ascending=False)
        '''
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
        '''
        Filters and sorts the observable table based on specified criteria.

        Parameters
        ----------
        info : dict
            A dictionary containing observable information.
        sortkey : str, optional
            The key by which to sort the observables. Default is `'name'`.
        ascending : bool, optional
            If `True`, sort in ascending order. If `False`, sort in descending order. Default is `True`.
        min_val : float, optional
            Minimum value for the `sortkey` to include an observable. Default is `None`.
        max_val : float, optional
            Maximum value for the `sortkey` to include an observable. Default is `None`.
        subset : callable, optional
            A function that takes a row and returns `True` if the row should be included. Default is `None`.
        max_rows : int, optional
            Maximum number of rows to include in the output. If the number of rows exceeds this value, the output is split into multiple tables. Default is `None`.

        Returns
        -------
        dict or list[dict]
            A filtered and sorted dictionary of observables, or a list of such dictionaries if the number of rows exceeds `max_rows`.
        '''
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
    '''
    Creates a new tree structure.

    Returns
    -------
    defaultdict
        A tree structure implemented as a nested defaultdict.

    Examples
    --------
    Create a new tree structure:

    >>> my_tree = tree()
    '''
    return defaultdict(tree)

def compute_pulls(log_likelihood, log_likelihood_sm):
    '''
    Computes the pulls with respect to the Standard Model and the central experimental likelihoods.

    Parameters
    ----------
    log_likelihood : np.ndarray
        The log-likelihood values for the observables.
    log_likelihood_sm : np.ndarray
        The Standard Model log-likelihood values for the observables.

    Returns
    -------
    tuple
        A tuple containing two np.ndarrays: the pulls with respect to the Standard Model and the pulls with respect to the central experimental values.

    Examples
    --------
    Compute the pulls for given log-likelihood values:

    >>> pulls_sm, pulls_exp = compute_pulls(log_likelihood, log_likelihood_sm)
    '''
    s = np.where(log_likelihood > log_likelihood_sm, -1, 1)
    pull_sm = s * np.sqrt(np.abs(-2 * (log_likelihood - log_likelihood_sm)))
    pull_exp = np.sqrt(np.abs(-2 * log_likelihood))
    return pull_sm, pull_exp
