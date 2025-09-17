from typing import List, Dict, Union, Tuple
from itertools import chain
from functools import partial
from jax import jit, numpy as jnp
import numpy as np
from numbers import Number
from operator import itemgetter
from wilson import Wilson, wcxf
import networkx as nx
from rgevolve.tools import get_wc_basis, reference_scale
from .observable_sector import ObservableSector
from .measurement import Measurement
from .custom_basis import CustomBasis
from .global_likelihood_point import GlobalLikelihoodPoint
from .theory_correlations import TheoryCorrelations
from .experimental_correlations import ExperimentalCorrelations
from jelli.utils.distributions import (
    logL_functions,
    logL_functions_summed,
    cov_coeff_to_cov_obs,
    logL_correlated_sectors,
    logL_correlated_sectors_summed,
    get_mode_and_uncertainty,
    LOG_ZERO,
)
from ..utils.par_helpers import get_wc_basis_from_wcxf
from collections import defaultdict
from .compiled_likelihood import CompiledLikelihood

class GlobalLikelihood():
    '''
    A class to represent the global likelihood.

    Parameters
    ----------
    eft : str, optional
        The EFT name (e.g., `SMEFT`, `WET`). Required if `custom_basis` is not provided.
    basis : str, optional
        The basis name (e.g., `Warsaw`, `JMS`). Required if `custom_basis` is not provided.
    custom_basis : str, optional
        The name of a custom basis defined using the `CustomBasis` class. Required if `eft` and `basis` are not provided.
    include_observable_sectors : list[str], optional
        A list of observable sector names to include in the likelihood. If not provided, all loaded observable sectors are included.
    exclude_observable_sectors : list[str], optional
        A list of observable sector names to exclude from the likelihood. If not provided, no sectors are excluded.
    include_measurements : list[str], optional
        A list of measurement names to include in the likelihood. If not provided, all loaded measurements are included.
    exclude_measurements : list[str], optional
        A list of measurement names to exclude from the likelihood. If not provided, no measurements are excluded.
    custom_likelihoods : dict[str, list[str]], optional
        A dictionary defining custom likelihoods. The keys are the names of the custom likelihoods, and the values are lists of observable names to include in each custom likelihood.

    Attributes
    ----------
    eft : str
        The EFT name (e.g., `SMEFT`, `WET`).
    basis : str
        The basis name (e.g., `Warsaw`, `JMS`).
    custom_basis : str
        The name of the custom basis defined using the `CustomBasis` class.
    observable_sectors_gaussian : list[str]
        The list of observable sector names containing observables with Gaussian theory uncertainties.
    observable_sectors_no_theory_uncertainty : list[str]
        The list of observable sector names containing observables with no theory uncertainty.
    basis_mode : str
        The basis mode, either `rgevolve`, `wcxf`, or `custom`.
    observable_sectors : list[str]
        The list of observable sector names included in the likelihood.
    parameter_basis : list[str]
        The list of parameter names in the basis.
    parameter_basis_split_re_im : list[Tuple[str, str or None]]
        The list of parameter names in the basis, split into real and imaginary parts. Each entry is a tuple where the first element is the parameter name and the second element is `R` for real parameters or `I` for imaginary parameters.
    include_measurements : dict[str, Measurement]
        The measurements included in the likelihood.
    observables_constrained : set[str]
        The set of observables constrained by the included measurements.
    observables_no_theory_uncertainty : list[str]
        The list of observables with no theory uncertainty.
    observables_gaussian : list[str]
        The list of observables with Gaussian theory uncertainties.
    observables_correlated : list[list[str]]
        The list of lists of observables in correlated observable sectors.
    prediction_data_no_theory_uncertainty : list[list[jnp.array]]
        The prediction data for observables with no theory uncertainty.
    prediction_function_no_theory_uncertainty : callable
        The prediction function for observables with no theory uncertainty.
    prediction_data_correlated : list[list[list[jnp.array]]]
        The prediction data for observables in correlated sectors.
    prediction_function_correlated : list[callable]
        The list of prediction functions for correlated observable sectors.
    custom_likelihoods_gaussian : dict[str, list[str]]
        The custom likelihoods containing observables with Gaussian theory uncertainties.
    custom_likelihoods_no_theory_uncertainty : dict[str, list[str]]
        The custom likelihoods containing observables with no theory uncertainty.
    likelihoods : list[str]
        The list of all likelihood names, including custom likelihoods and 'global'.
    constraints_no_theory_uncertainty : dict
        The constraints for observables with no theory uncertainty.
    constraints_no_theory_uncertainty_no_corr : dict
        The constraints for observables with no theory uncertainty, neglecting experimental correlations (used for observable table).
    selector_matrix_no_th_unc_univariate : jnp.array
        The selector matrix mapping observables with no theory uncertainty to likelihoods for univariate distributions.
    selector_matrix_no_th_unc_multivariate : jnp.array
        The selector matrix mapping unique multivariate normal contributions to likelihoods for observables with no theory uncertainty.
    constraints_correlated_par_indep_cov : dict
        The constraints for observables in correlated sectors with parameter-independent covariance.
    constraints_correlated_par_dep_cov : dict
        The constraints for observables in correlated sectors with parameter-dependent covariance.
    selector_matrix_correlated : List[jnp.array]
        The selector matrices mapping unique multivariate normal contributions to likelihoods for observables in correlated sectors.
    sm_log_likelihood_summed : jnp.array
        The Standard Model log-likelihood summed over all observables.
    sm_log_likelihood_correlated : jnp.array
        The Standard Model log-likelihood values for correlated observables.
    sm_log_likelihood_correlated_no_corr : jnp.array
        The Standard Model log-likelihood values for correlated observables, neglecting correlations (used for observable table).
    sm_log_likelihood_no_theory_uncertainty : jnp.array
        The Standard Model log-likelihood values for observables with no theory uncertainty.
    sm_log_likelihood_no_theory_uncertainty_no_corr : jnp.array
        The Standard Model log-likelihood values for observables with no theory uncertainty, neglecting correlations (used for observable table).
    experimental_values_no_theory_uncertainty : dict[str, list[float]]
        A dictionary mapping observable names to their experimental values and uncertainties for observables with no theory uncertainty (used for observable table).
    _observables_per_likelihood_no_theory_uncertainty : dict[str, list[str]]
        A dictionary mapping likelihood names to lists of observables with no theory uncertainty.
    _observables_per_likelihood_correlated : dict[str, list[str]]
        A dictionary mapping likelihood names to lists of observables in correlated sectors.
    _likelihood_indices_no_theory_uncertainty : jnp.array
        The indices of the likelihoods with no theory uncertainty in the full likelihood list.
    _likelihood_indices_correlated : jnp.array
        The indices of the correlated likelihoods in the full likelihood list.
    _likelihood_indices_global : jnp.array
        The indices of the likelihoods included in the global likelihood (i.e., not custom likelihoods).
    _reference_scale : float
        The reference scale for the likelihood.
    _indices_mvn_not_custom : jnp.array
        The indices of multivariate normal contributions not included in custom likelihoods.
    _log_likelihood_point_function : callable
        The JIT-compiled function to compute the information needed for `GlobalLikelihoodPoint` instances.
    _log_likelihood_point : callable
        A partial function wrapping `_log_likelihood_point_function` with fixed arguments.
    _obstable : callable
        The JIT-compiled function to compute the observable table information.
    _cache_compiled_likelihood : dict
        A cache for `CompiledLikelihood` instances to avoid redundant computations.

    Methods
    -------
    load(path: str) -> None
        Initializes `ObservableSector`, `Measurement`, `TheoryCorrelations`, and `ExperimentalCorrelations` classes by loading data from the specified path.
    get_negative_log_likelihood(par_list: List[Tuple[str, str]], likelihood: Union[str, Tuple[str, ...]], par_dep_cov: bool) -> Tuple[Callable, List]
        Returns a function to compute the negative log-likelihood for given parameters and likelihood.
    parameter_point(*args, par_dep_cov: bool = False) -> GlobalLikelihoodPoint
        Returns a `GlobalLikelihoodPoint` instance for the specified parameter values.
    get_compiled_likelihood(par_list: List[Tuple[str, str]], likelihood: Union[str, Tuple[str, ...]], par_dep_cov: bool = False) -> CompiledLikelihood
        Returns an instance of `CompiledLikelihood` for the specified parameters and likelihood.
    plot_data_2d(par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20, par_dep_cov=False) -> Dict
        Computes a grid of chi-squared values over a 2D parameter space for plotting. Returns a dictionary containing the parameter grid and the corresponding chi-squared values.
    _get_observable_sectors(include_observable_sectors, exclude_observable_sectors) -> Tuple[List[str], List[str], str]
        Determines the observable sectors to include in the likelihood based on inclusion/exclusion lists.
    _get_observable_sectors_correlated() -> Tuple[List[List[str]], List[List[jnp.array]], List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]
        Determines and returns useful information about correlated observable sectors.
    _get_custom_likelihoods(custom_likelihoods) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Processes custom likelihoods.
    _get_observables_per_likelihood() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Constructs dictionaries mapping likelihood names to lists of observables for both no theory uncertainty and correlated sectors.
    _get_prediction_function_gaussian(observable_sectors_gaussian) -> Callable
        Returns a prediction function for the Gaussian observable sectors.
    _get_prediction_function_no_theory_uncertainty() -> Callable
        Returns a prediction function for observables with no theory uncertainty.
    _get_constraints_no_theory_uncertainty(observables, observable_lists_per_likelihood=None) -> Tuple[Dict, Dict, jnp.array, jnp.array, jnp.array]
        Returns the constraints and selector matrices for observables with no theory uncertainty.
    _get_constraints_correlated() -> Tuple[Dict, Dict, List[jnp.array]]
        Returns the constraints and selector matrices for correlated observable sectors.
    _get_log_likelihood_point_function() -> Callable
        Returns a JIT-compiled function to compute the information needed for `GlobalLikelihoodPoint` instances.
    _get_obstable_function() -> Callable
        Returns a JIT-compiled function to compute the observable table information.
    _get_parameter_basis() -> Tuple[Dict, Dict]
        Determines the parameter basis and splits parameters into real and imaginary parts.
    _get_par_array(par_dict: Dict) -> jnp.array
        Converts a parameter dictionary into a JAX array.
    _get_reference_scale() -> float
        Determines the reference scale for the likelihood.

    Examples
    --------
    Load all observable sectors, measurements, and correlations from the specified path:

    >>> GlobalLikelihood.load('path/to/data')

    Create a global likelihood instance for the SMEFT in the Warsaw basis, including all observable sectors and measurements:

    >>> gl = GlobalLikelihood(eft='SMEFT', basis='Warsaw')

    Create a global likelihood instance for a custom basis named 'my_basis', including only specific observable sectors and measurements:

    >>> gl = GlobalLikelihood(custom_basis='my_basis', include_observable_sectors=['sector1', 'sector2'], include_measurements=['measurement1', 'measurement2'])

    Create a global likelihood instance for the SMEFT in the Warsaw basis, defining a custom likelihood that includes specific observables:

    >>> custom_likelihoods = {'my_likelihood': ['observable1', 'observable2']}
    >>> gl = GlobalLikelihood(eft='SMEFT', basis='Warsaw', custom_likelihoods=custom_likelihoods)

    Define a `GlobalLikelihoodPoint` instance for specific parameter values at the scale of 1000.0 GeV:

    >>> def par_func(x, y):
    ...     return {'lq1_1111': x, 'lq3_1111': y}
    >>> glp = gl.parameter_point(par_func(1e-8, 1e-8), 1000.0)

    Obtain the 2D chi-squared grid for two parameters over specified ranges:

    >>> plot_data = gl.plot_data_2d(par_func, scale=1000.0, x_min=-1e-8, x_max=1e-8, y_min=-1e-8, y_max=1e-8, steps=50)

    Get the negative log-likelihood function and data for specific parameters and a likelihood:

    >>> negative_log_likelihood, log_likelihood_data = gl.get_negative_log_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', par_dep_cov=False)

    Get an instance of `CompiledLikelihood` for specific parameters and a likelihood:

    >>> compiled_likelihood = gl.get_compiled_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', par_dep_cov=False)

    Access the parameter basis:

    >>> parameter_basis = gl.parameter_basis
    >>> parameter_basis_split_re_im = gl.parameter_basis_split_re_im

    Access the basis mode:

    >>> basis_mode = gl.basis_mode

    Access the observables included in the likelihood:

    >>> observables_gaussian = gl.observables_gaussian
    >>> observables_no_theory_uncertainty = gl.observables_no_theory_uncertainty

    '''

    def __init__(
        self,
        eft=None,
        basis=None,
        custom_basis=None,
        include_observable_sectors=None,
        exclude_observable_sectors=None,
        include_measurements=None,
        exclude_measurements=None,
        custom_likelihoods=None,
    ):
        '''
        Initialize the GlobalLikelihood instance.

        Parameters
        ----------
        eft : str, optional
            The EFT name (e.g., `SMEFT`, `WET`). Required if `custom_basis` is not provided.
        basis : str, optional
            The basis name (e.g., `Warsaw`, `JMS`). Required if `custom_basis` is not provided.
        custom_basis : str, optional
            The name of a custom basis defined using the `CustomBasis` class. Required if `eft` and `basis` are not provided.
        include_observable_sectors : list[str], optional
            A list of observable sector names to include in the likelihood. If not provided, all loaded observable sectors are included.
        exclude_observable_sectors : list[str], optional
            A list of observable sector names to exclude from the likelihood. If not provided, no sectors are excluded.
        include_measurements : list[str], optional
            A list of measurement names to include in the likelihood. If not provided, all loaded measurements are included.
        exclude_measurements : list[str], optional
            A list of measurement names to exclude from the likelihood. If not provided, no measurements are excluded.
        custom_likelihoods : dict[str, list[str]], optional
            A dictionary defining custom likelihoods. The keys are the names of the custom likelihoods, and the values are lists of observable names to include in each custom likelihood.

        Returns
        -------
        None

        Examples
        --------
        Initialize a global likelihood instance for the SMEFT in the Warsaw basis, including all observable sectors and measurements:

        >>> gl = GlobalLikelihood(eft='SMEFT', basis='Warsaw')

        Initialize a global likelihood instance for a custom basis named 'my_basis', including only specific observable sectors and measurements:

        >>> gl = GlobalLikelihood(custom_basis='my_basis', include_observable_sectors=['sector1', 'sector2'], include_measurements=['measurement1', 'measurement2'])

        Initialize a global likelihood instance for the SMEFT in the Warsaw basis, defining a custom likelihood that includes specific observables:

        >>> custom_likelihoods = {'my_likelihood': ['observable1', 'observable2']}
        >>> gl = GlobalLikelihood(eft='SMEFT', basis='Warsaw', custom_likelihoods=custom_likelihoods)
        '''

        if custom_basis is not None:
            if eft is not None or basis is not None:
                raise ValueError("Please provide either `custom_basis`, or both `eft` and `basis`, but not both.")
        elif eft is not None and basis is None or basis is not None and eft is None:
            raise ValueError("Please provide the `eft` when using the `basis` and vice versa.")


        # define attributes from arguments

        self.eft = eft
        self.basis = basis
        self.custom_basis = custom_basis


        # get names of all observable sectors and the basis mode, basis parameters, and reference scale

        (
            self.observable_sectors_gaussian,
            self.observable_sectors_no_theory_uncertainty,
            self.basis_mode
        ) = self._get_observable_sectors(
            include_observable_sectors,
            exclude_observable_sectors
        )
        self.observable_sectors = self.observable_sectors_gaussian + self.observable_sectors_no_theory_uncertainty
        self.parameter_basis_split_re_im, self.parameter_basis = self._get_parameter_basis()
        self._reference_scale = self._get_reference_scale()

        # get all measurements
        observables_all = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors
        ))
        self.include_measurements = Measurement.get_measurements(
            observables=observables_all,
            include_measurements=include_measurements,
            exclude_measurements=exclude_measurements,
        )
        self.observables_constrained = set(chain.from_iterable(
            measurement.constrained_observables
            for measurement in self.include_measurements.values()
        ))

        # define attributes for observable sectors with no theory uncertainty

        self.observables_no_theory_uncertainty = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        ))
        self.prediction_data_no_theory_uncertainty = [
            ObservableSector.get(observable_sector).get_prediction_data(self.eft, self.basis)
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        ]
        self.prediction_function_no_theory_uncertainty = self._get_prediction_function_no_theory_uncertainty()


        # define attributes for correlated observable sectors

        (
            self.observable_sectors_correlated,
            self.cov_coeff_th_scaled,
            self.cov_exp_scaled,
            self.exp_central_scaled,
            self.std_sm_exp,
            self.std_exp,
        ) = self._get_observable_sectors_correlated()

        self.observables_correlated = [
            list(chain.from_iterable(
                ObservableSector.get(observable_sector).observable_names
                for observable_sector in observable_sectors
            ))
            for observable_sectors in self.observable_sectors_correlated
        ]
        self.prediction_data_correlated = [
            [
                ObservableSector.get(observable_sector).get_prediction_data(self.eft, self.basis)
                for observable_sector in observable_sectors
            ]
            for observable_sectors in self.observable_sectors_correlated
        ]
        self.prediction_function_correlated = [
            self._get_prediction_function_gaussian(observable_sectors)
            for observable_sectors in self.observable_sectors_correlated
        ]

        self.observables_gaussian = list(chain.from_iterable(
            self.observables_correlated
            ))

        self.custom_likelihoods_gaussian, self.custom_likelihoods_no_theory_uncertainty = self._get_custom_likelihoods(custom_likelihoods)
        self._observables_per_likelihood_no_theory_uncertainty, self._observables_per_likelihood_correlated = self._get_observables_per_likelihood()

        _likelihoods_no_theory_uncertainty = sorted(self._observables_per_likelihood_no_theory_uncertainty.keys())
        _likelihoods_correlated = sorted(self._observables_per_likelihood_correlated.keys())
        _likelihoods_custom = sorted(set(self.custom_likelihoods_gaussian.keys()) | set(self.custom_likelihoods_no_theory_uncertainty.keys()))
        _likelihoods = _likelihoods_correlated + _likelihoods_no_theory_uncertainty + _likelihoods_custom

        self._observables_per_likelihood_no_theory_uncertainty.update(self.custom_likelihoods_no_theory_uncertainty)
        self._observables_per_likelihood_correlated.update(self.custom_likelihoods_gaussian)
        self._likelihood_indices_no_theory_uncertainty = jnp.array([
            _likelihoods.index(likelihood)
            for likelihood in list(self._observables_per_likelihood_no_theory_uncertainty.keys())
        ], dtype=int)
        self._likelihood_indices_correlated = jnp.array([
            _likelihoods.index(likelihood)
            for likelihood in list(self._observables_per_likelihood_correlated.keys())
        ], dtype=int)

        # add global likelihood
        self._likelihood_indices_global = jnp.array([
            i for i, likelihood in enumerate(_likelihoods)
            if likelihood not in (
                set(self.custom_likelihoods_gaussian) | set(self.custom_likelihoods_no_theory_uncertainty)
            )
        ], dtype=int)
        self.likelihoods = _likelihoods + ['global']

        (
            self.constraints_no_theory_uncertainty,
            self.constraints_no_theory_uncertainty_no_corr,
            self.selector_matrix_no_th_unc_univariate,
            self.selector_matrix_no_th_unc_multivariate,
            self._indices_mvn_not_custom,
        ) = self._get_constraints_no_theory_uncertainty(
            self.observables_no_theory_uncertainty,
            list(self._observables_per_likelihood_no_theory_uncertainty.values())
        )

        (
            self.constraints_correlated_par_indep_cov,
            self.constraints_correlated_par_dep_cov,
            self.selector_matrix_correlated,
        ) = self._get_constraints_correlated()

        self._log_likelihood_point_function = self._get_log_likelihood_point_function()
        self._log_likelihood_point = partial(
            self._log_likelihood_point_function,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            prediction_data_correlated=self.prediction_data_correlated,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty,
            constraints_correlated_par_indep_cov=self.constraints_correlated_par_indep_cov,
            constraints_correlated_par_dep_cov=self.constraints_correlated_par_dep_cov,
            selector_matrix_no_th_unc_univariate=self.selector_matrix_no_th_unc_univariate,
            selector_matrix_no_th_unc_multivariate=self.selector_matrix_no_th_unc_multivariate,
            selector_matrix_correlated=self.selector_matrix_correlated,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_correlated=self._likelihood_indices_correlated,
            likelihood_indices_global=self._likelihood_indices_global,
        )
        (
            sm_prediction_no_theory_uncertainty,
            sm_prediction_correlated,
            sm_log_likelihood_no_th_unc_univariate,
            sm_log_likelihood_no_th_unc_multivariate,
            sm_log_likelihood_correlated,
            self.sm_log_likelihood_summed,
            std_sm_exp_correlated_scaled,
        ) = self._log_likelihood_point(
            self._get_par_array({}),
            self._reference_scale,
            par_dep_cov=False,
        )

        self._obstable = partial(
            self._get_obstable_function(),
            constraints_no_theory_uncertainty_no_corr=self.constraints_no_theory_uncertainty_no_corr,
            indices_mvn_not_custom=self._indices_mvn_not_custom,
            exp_central_scaled=self.exp_central_scaled,
            std_sm_exp=self.std_sm_exp,
        )
        (
            sm_log_likelihood_no_th_unc_multivariate,
            sm_log_likelihood_no_th_unc_multivariate_no_corr,
            self.sm_log_likelihood_correlated,
            self.sm_log_likelihood_correlated_no_corr,
            _,
            _,
        ) = self._obstable(
            sm_prediction_no_theory_uncertainty,
            sm_prediction_correlated,
            sm_log_likelihood_no_th_unc_multivariate,
            sm_log_likelihood_correlated,
            std_sm_exp_correlated_scaled,
        )
        self.sm_log_likelihood_no_theory_uncertainty = sm_log_likelihood_no_th_unc_univariate + sm_log_likelihood_no_th_unc_multivariate
        self.sm_log_likelihood_no_theory_uncertainty_no_corr = sm_log_likelihood_no_th_unc_univariate + sm_log_likelihood_no_th_unc_multivariate_no_corr

        combined_constraints = Measurement.get_combined_constraints(
            self.observables_no_theory_uncertainty
        )
        experimental_values = {}
        for dist_type, dist_info in combined_constraints.items():
            observable_indices = dist_info['observable_indices']
            mode, uncertainty = get_mode_and_uncertainty(dist_type, dist_info)
            experimental_values.update({
                self.observables_no_theory_uncertainty[ind]: [mode[i], uncertainty[i]]
                for i, ind in enumerate(observable_indices)
            })
        self.experimental_values_no_theory_uncertainty = experimental_values

        self._cache_compiled_likelihood = {}

    @classmethod
    def load(cls, path):
        '''
        Initialize `ObservableSector`, `Measurement`, `TheoryCorrelations`, and `ExperimentalCorrelations` classes by loading data from the specified path.

        Parameters
        ----------
        path : str
            The path to the directory containing the data files.

        Returns
        -------
        None

        Examples
        --------

        Load all observable sectors, measurements, and correlations from the specified path:

        >>> GlobalLikelihood.load('path/to/data')
        '''
        # load all observable sectors
        ObservableSector.load(path)
        # load all measurements
        Measurement.load(path)
        # load all theory correlations
        TheoryCorrelations.load(path)
        # load all experimental correlations
        ExperimentalCorrelations.load()

    def _get_observable_sectors(self, include_observable_sectors, exclude_observable_sectors):
        '''
        Determines the observable sectors to include in the likelihood based on inclusion/exclusion lists.

        Parameters
        ----------
        include_observable_sectors : list[str] or None
            A list of observable sector names to include in the likelihood. If None, all loaded observable sectors are included.
        exclude_observable_sectors : list[str] or None
            A list of observable sector names to exclude from the likelihood. If None, no sectors are excluded.

        Returns
        -------
        observable_sectors_gaussian : list[str]
            The list of observable sector names containing observables with Gaussian theory uncertainties.
        observable_sectors_no_theory_uncertainty : list[str]
            The list of observable sector names containing observables with no theory uncertainty.
        basis_mode : str
            The basis mode, either `rgevolve`, `wcxf`, or `custom`.
        '''
        if include_observable_sectors is not None and exclude_observable_sectors is not None:
            raise ValueError("Please provide either `include_observable_sectors` or `exclude_observable_sectors`, not both.")
        available_observable_sectors = set(ObservableSector.get_all_names(eft=self.eft, basis=self.basis, custom_basis=self.custom_basis))
        if include_observable_sectors is not None:
            if set(include_observable_sectors)-available_observable_sectors:
                raise ValueError(f"Observable sectors {set(include_observable_sectors)-available_observable_sectors} provided in `include_observable_sectors` but not found in loaded observable sectors")
            observable_sectors = sorted(
                include_observable_sectors
            )
        elif exclude_observable_sectors is not None:
            if set(exclude_observable_sectors)-available_observable_sectors:
                raise ValueError(f"Observable sectors {set(exclude_observable_sectors)-available_observable_sectors} provided in `exclude_observable_sectors` but not found in loaded observable sectors")
            observable_sectors = sorted(
                available_observable_sectors - set(exclude_observable_sectors)
            )
        else:
            observable_sectors = sorted(available_observable_sectors)
        if observable_sectors:
            basis_mode = ObservableSector.get(observable_sectors[0]).basis_mode
            if basis_mode in ['wcxf', 'custom']:
                scales = set(
                    ObservableSector.get(observable_sector).scale
                    for observable_sector in observable_sectors
                )
                if len(scales) > 1:
                    raise ValueError(
                        f"Observable sectors for basis {self.custom_basis or (self.eft, self.basis)} are defined at different scales. Please use `include_observable_sectors` or `exclude_observable_sectors` to select observable sectors at the same scale."
                    )
        observable_sectors_gaussian = []
        observable_sectors_no_theory_uncertainty = []
        for observable_sector in observable_sectors:
            if ObservableSector.get(observable_sector).observable_uncertainties is None:
                observable_sectors_no_theory_uncertainty.append(observable_sector)
            else:
                observable_sectors_gaussian.append(observable_sector)
        return observable_sectors_gaussian, observable_sectors_no_theory_uncertainty, basis_mode

    def _get_observable_sectors_correlated(self):
        '''
        Determines and returns useful information about correlated observable sectors.

        Returns
        -------
        observable_sectors_correlated : list[list[str]]
            The list of lists of observable sector names in correlated groups.
        cov_coeff_th_scaled : list[list[list[jnp.array]]]
            The list of lists of theory correlation coefficient matrices for each correlated group, scaled by the combined SM and experimental uncertainties.
        cov_exp_scaled : list[jnp.array]
            The list of experimental covariance matrices for each correlated group, scaled by the combined SM and experimental uncertainties.
        exp_central_scaled : list[jnp.array]
            The list of experimental central values for each correlated group, scaled by the combined SM and experimental uncertainties.
        std_sm_exp : list[jnp.array]
            The list of combined SM and experimental uncertainties for each correlated group.
        std_exp_list : list[jnp.array]
            The list of experimental uncertainties for each correlated group.
        '''

        # get correlations for all gaussian observable sectors

        correlations_th =  []
        correlations_exp =  []
        for i, row_sector in enumerate(self.observable_sectors_gaussian):
            row_th = []
            row_exp = []
            for j, col_sector in enumerate(self.observable_sectors_gaussian[:i+1]):
                obs_row = ObservableSector.get(row_sector).observable_names
                obs_col = ObservableSector.get(col_sector).observable_names
                row_th.append(TheoryCorrelations.get_data(obs_row, obs_col))
                row_exp.append(ExperimentalCorrelations.get_data('correlations', self.include_measurements, obs_row, obs_col))
            correlations_th.append(row_th)
            correlations_exp.append(row_exp)


        # find connected components of the correlation graph

        G = nx.Graph()
        G.add_nodes_from(self.observable_sectors_gaussian)
        for i, name_i in enumerate(self.observable_sectors_gaussian):
            for j, name_j in enumerate(self.observable_sectors_gaussian[:i+1]):
                if correlations_th[i][j] is not None or correlations_exp[i][j] is not None:
                    G.add_edge(name_i, name_j)
        components = list(nx.connected_components(G))
        components = [sorted(list(group)) for group in components]
        components = sorted(components, key=lambda c: self.observable_sectors_gaussian.index(c[0]))
        observable_sectors_correlated = components


        # get combined sm and exp standard deviations and scaled uncertainties for connected components

        std_th_scaled = []
        std_exp_scaled = []
        std_sm_exp = []
        exp_central_scaled = []
        std_exp_list = []
        for group in components:
            sub_std_th_scaled = []
            sub_std_exp_scaled = []
            sub_std_sm_exp = []
            sub_exp_central_scaled = []
            sub_std_exp = []
            for i, row_sector in enumerate(group):
                obs_row = ObservableSector.get(row_sector).observable_names
                std_exp = ExperimentalCorrelations.get_data('uncertainties', self.include_measurements, obs_row)
                exp_central = ExperimentalCorrelations.get_data('central', self.include_measurements, obs_row)
                std_th = ObservableSector.get(row_sector).observable_uncertainties
                std_sm = ObservableSector.get(row_sector).observable_uncertainties_SM
                _std_sm_exp = std_exp * np.sqrt(1 + (std_sm / std_exp)**2) # combined sm + exp uncertainty
                sub_std_th_scaled.append(std_th/_std_sm_exp)
                sub_std_exp_scaled.append(std_exp/_std_sm_exp)
                sub_std_sm_exp.append(_std_sm_exp)
                sub_exp_central_scaled.append(exp_central/_std_sm_exp)
                sub_std_exp.append(std_exp)
            std_th_scaled.append(sub_std_th_scaled)
            std_exp_scaled.append(sub_std_exp_scaled)
            std_sm_exp.append(jnp.array(np.concatenate(sub_std_sm_exp)))
            exp_central_scaled.append(jnp.array(np.concatenate(sub_exp_central_scaled)))
            std_exp_list.append(jnp.array(np.concatenate(sub_std_exp)))


        # get scaled covariance matrices for connected components

        cov_coeff_th_scaled = []
        cov_exp_scaled = []
        for k, group in enumerate(components):
            sub_th = []
            sub_exp = []
            for i, row_sector in enumerate(group):
                row_th = []
                row_exp = []
                for j, col_sector in enumerate(group[:i+1]):
                    obs_row = ObservableSector.get(row_sector).observable_names
                    obs_col = ObservableSector.get(col_sector).observable_names
                    row_th.append(TheoryCorrelations.get_cov_scaled(
                        self.include_measurements, obs_row, obs_col, std_th_scaled[k][i], std_th_scaled[k][j]
                    ))
                    row_exp.append(ExperimentalCorrelations.get_cov_scaled(
                        self.include_measurements, obs_row, obs_col, std_exp_scaled[k][i], std_exp_scaled[k][j]
                    ))
                sub_th.append(row_th)
                sub_exp.append(row_exp)
            cov_coeff_th_scaled.append(sub_th)

            n_sectors = len(sub_exp)
            cov_exp = np.empty((n_sectors, n_sectors), dtype=object).tolist()
            for i in range(n_sectors):
                for j in range(n_sectors):
                    if i >= j:
                        cov_exp[i][j] = sub_exp[i][j]
                    else:
                        shape = sub_exp[j][i].shape
                        cov_exp[i][j] = np.zeros((shape[1], shape[0]))
            cov_exp_tril = np.tril(np.block(cov_exp))
            sub_exp = cov_exp_tril + cov_exp_tril.T - np.diag(np.diag(cov_exp_tril))
            cov_exp_scaled.append(jnp.array(sub_exp))

        return (
            observable_sectors_correlated,
            cov_coeff_th_scaled,
            cov_exp_scaled,
            exp_central_scaled,
            std_sm_exp,
            std_exp_list,
        )

    def _get_custom_likelihoods(self, custom_likelihoods):
        '''
        Processes custom likelihoods.

        Parameters
        ----------
        custom_likelihoods : dict[str, list[str]] or None
            A dictionary defining custom likelihoods. The keys are the names of the custom likelihoods, and the values are lists of observable names to include in each custom likelihood.

        Returns
        -------
        likelihoods_gaussian : dict[str, list[str]]
            A dictionary mapping custom likelihood names to lists of observables with Gaussian theory uncertainties.
        likelihoods_no_theory_uncertainty : dict[str, list[str]]
            A dictionary mapping custom likelihood names to lists of observables with no theory uncertainty.
        '''
        if custom_likelihoods is None:
            return {}, {}
        if not isinstance(custom_likelihoods, dict) or not all([isinstance(k, str) and isinstance(v, list) for k, v in custom_likelihoods.items()]):
            raise ValueError("The custom_likelihoods argument should be a dictionary with string names of custom likelihoods as keys and lists of observable names as values.")

        likelihoods_gaussian = {}
        likelihoods_no_theory_uncertainty = {}

        for name, observables in custom_likelihoods.items():
            observables_gaussian = set()
            observables_no_theory_uncertainty = set()
            invalid_observables = set()
            for observable in observables:
                if observable in self.observables_gaussian:
                    observables_gaussian.add(observable)
                elif observable in self.observables_no_theory_uncertainty:
                    observables_no_theory_uncertainty.add(observable)
                else:
                    invalid_observables.add(observable)
            if invalid_observables:
                raise ValueError(
                    f"Custom likelihood '{name}' contains observables not found in the loaded observable sectors: {sorted(invalid_observables)}"
                )
            if observables_gaussian:
                likelihoods_gaussian[f'custom_{name}'] = sorted(observables_gaussian)
            if observables_no_theory_uncertainty:
                likelihoods_no_theory_uncertainty[f'custom_{name}'] = sorted(observables_no_theory_uncertainty)

        return likelihoods_gaussian, likelihoods_no_theory_uncertainty

    def _get_observables_per_likelihood(self):
        '''
        Constructs dictionaries mapping likelihood names to lists of observables for both no theory uncertainty and correlated sectors.

        Returns
        -------
        observables_per_likelihood_no_theory_uncertainty : dict[str, list[str]]
            A dictionary mapping likelihood names to lists of observables with no theory uncertainty.
        observables_per_likelihood_correlated : dict[str, list[str]]
            A dictionary mapping likelihood names to lists of observables with Gaussian theory uncertainties.
        '''

        observables_per_likelihood_no_theory_uncertainty = {
            observable_sector: ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        }

        observables_per_likelihood_correlated = {
            tuple(observable_sectors): self.observables_correlated[i]
            for i, observable_sectors in enumerate(self.observable_sectors_correlated)
            }

        return observables_per_likelihood_no_theory_uncertainty, observables_per_likelihood_correlated

    def _get_prediction_function_gaussian(self, observable_sectors_gaussian):
        '''
        Returns a prediction function for the Gaussian observable sectors.

        Parameters
        ----------
        observable_sectors_gaussian : list[str]
            A list of observable sector names containing observables with Gaussian theory uncertainties.

        Returns
        -------
        prediction : Callable
            A function that takes a parameter array, scale, and prediction data, and returns the polynomial predictions and parameter monomials.
        '''

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in observable_sectors_gaussian
        ]

        def prediction(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data: List[List[jnp.array]]
        ) -> jnp.array:
            polynomial_predictions = [jnp.empty(0)]
            par_monomials = []
            for prediction_function, data in zip(prediction_functions, prediction_data):
                polynomial_prediction, par_monomial = prediction_function(
                    par_array, scale, data
                )
                polynomial_predictions.append(polynomial_prediction)
                par_monomials.append(par_monomial)
            polynomial_predictions = jnp.concatenate(polynomial_predictions, axis=-1)
            return polynomial_predictions, par_monomials

        return prediction

    def _get_prediction_function_no_theory_uncertainty(self):
        '''
        Returns a prediction function for observables with no theory uncertainty.

        Returns
        -------
        prediction : Callable
            A function that takes a parameter array, scale, and prediction data, and returns the polynomial predictions for observables with no theory uncertainty.
        '''

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in self.observable_sectors_no_theory_uncertainty
        ]
        def prediction(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data: List[List[jnp.array]]
        ) -> jnp.array:
            polynomial_predictions = [jnp.empty(0)]
            for prediction_function, data in zip(prediction_functions, prediction_data):
                polynomial_predictions.append(
                    prediction_function(par_array, scale, data)[0]
                )
            polynomial_predictions = jnp.concatenate(polynomial_predictions, axis=-1)
            return polynomial_predictions


        return prediction

    def _get_constraints_no_theory_uncertainty(self, observables, observable_lists_per_likelihood=None):
        '''
        Returns the constraints and selector matrices for observables with no theory uncertainty.

        Parameters
        ----------
        observables : list[str]
            A list of observable names with no theory uncertainty.
        observable_lists_per_likelihood : list[list[str]] or None
            A list of lists of observable names for each likelihood.

        Returns
        -------
        constraint_dict : dict
            A dictionary containing the constraints for different distribution types.
        constraint_no_corr : list or None
            A list containing the multivariate normal distribution constraints neglecting correlations, or None if no such constraints exist.
        selector_matrix_univariate : jnp.array
            A selector matrix for univariate distributions, with shape `(n_likelihoods, n_observables)`.
        selector_matrix_multivariate : jnp.array
            A selector matrix for multivariate normal distributions, with shape `(n_likelihoods, n_distributions)`.
        indices_mvn_not_custom : jnp.array
            Indices of multivariate normal distributions that contribute to non-custom likelihoods.
        '''

        constraint_dict = {}

        constraints = Measurement.get_constraints(
            observables,
            include_measurements=self.include_measurements,
            distribution_types=[
                'NumericalDistribution',
                'NormalDistribution',
                'HalfNormalDistribution',
                'GammaDistributionPositive',
                'MultivariateNormalDistribution',
            ]
        )

        # numerical distribution
        if 'NumericalDistribution' in constraints:
            constraint_dict['NumericalDistribution'] = [
                jnp.asarray(constraints['NumericalDistribution']['observable_indices']),
                jnp.asarray(constraints['NumericalDistribution']['x']),
                jnp.asarray(constraints['NumericalDistribution']['log_y']),
            ]

        # normal distribution
        if 'NormalDistribution' in constraints:
            constraint_dict['NormalDistribution'] = [
                jnp.asarray(constraints['NormalDistribution']['observable_indices']),
                jnp.asarray(constraints['NormalDistribution']['central_value']),
                jnp.asarray(constraints['NormalDistribution']['standard_deviation']),
            ]

        # half normal distribution
        if 'HalfNormalDistribution' in constraints:
            constraint_dict['HalfNormalDistribution'] = [
                jnp.asarray(constraints['HalfNormalDistribution']['observable_indices']),
                jnp.asarray(constraints['HalfNormalDistribution']['standard_deviation']),
            ]

        # gamma distribution positive
        if 'GammaDistributionPositive' in constraints:
            constraint_dict['GammaDistributionPositive'] = [
                jnp.asarray(constraints['GammaDistributionPositive']['observable_indices']),
                jnp.asarray(constraints['GammaDistributionPositive']['a']),
                jnp.asarray(constraints['GammaDistributionPositive']['loc']),
                jnp.asarray(constraints['GammaDistributionPositive']['scale']),
            ]

        # MVN constraints, neglecting correlations
        if 'MultivariateNormalDistribution' in constraints:
            constraint_no_corr = [
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['observable_indices'])),
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['central_value'])),
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['standard_deviation'])),
            ]
        else:
            constraint_no_corr = None

        if observable_lists_per_likelihood is not None:  # if not only correlated likelihoods
            # selector matrix for univariate distributions
            selector_matrix_univariate = jnp.array([
                np.isin(observables, likelihood_observables).astype(float)
                for likelihood_observables in observable_lists_per_likelihood
            ])
        else:
            selector_matrix_univariate = jnp.zeros((0, len(observables)), dtype=float)

        # multivariate normal distribution

        _observable_lists_per_likelihood = observable_lists_per_likelihood or [observables]
        # Collect all unique MVN blocks into this dict
        unique_mvnd_blocks = {}

        # For each likelihood, keep track of which MVNs it uses (by key)
        mvnd_keys_per_likelihood = [[] for _ in _observable_lists_per_likelihood]

        # Loop over all likelihood definitions
        for i, observable_list in enumerate(_observable_lists_per_likelihood):

            mvnd_block_data = Measurement.get_constraints(
                observable_list,
                include_measurements=self.include_measurements,
                observables_for_indices=observables,
                distribution_types=['MultivariateNormalDistribution'],
            )['MultivariateNormalDistribution']

            for j in range(len(mvnd_block_data['measurement_name'])):
                mvnd_entry = {k: mvnd_block_data[k][j] for k in mvnd_block_data.keys()}
                mvnd_key = (mvnd_entry['measurement_name'], tuple(mvnd_entry['observables']))
                unique_mvnd_blocks[mvnd_key] = mvnd_entry
                mvnd_keys_per_likelihood[i].append(mvnd_key)

        # Final ordered list of all unique MVN blocks
        all_mvnd_keys = list(unique_mvnd_blocks.keys())

        n_likelihoods = len(mvnd_keys_per_likelihood)
        n_contributions = len(all_mvnd_keys)

        # Map MVND key to its index in all_mvnd_keys for fast lookup
        mvnd_key_to_index = {key: i for i, key in enumerate(all_mvnd_keys)}

        # Construct the logpdf input data from the unique MVNs
        if all_mvnd_keys:
            constraint_dict['MultivariateNormalDistribution'] = [
                [jnp.asarray(unique_mvnd_blocks[k]['observable_indices']) for k in all_mvnd_keys],
                [jnp.asarray(unique_mvnd_blocks[k]['central_value']) for k in all_mvnd_keys],
                [jnp.asarray(unique_mvnd_blocks[k]['standard_deviation']) for k in all_mvnd_keys],
                [jnp.asarray(unique_mvnd_blocks[k]['inverse_correlation']) for k in all_mvnd_keys],
            ]
            # Create selector matrix (n_likelihoods x n_contributions)
            selector_matrix_multivariate = np.zeros((n_likelihoods, n_contributions))
            for i, mvnd_keys in enumerate(mvnd_keys_per_likelihood):
                for key in mvnd_keys:
                    selector_matrix_multivariate[i, mvnd_key_to_index[key]] = 1.0
            selector_matrix_multivariate = jnp.array(selector_matrix_multivariate)
        else:
            selector_matrix_multivariate = jnp.zeros((n_likelihoods, 1), dtype=float)

        # Get indices of MVNs that contribute to non-custom likelihoods
        n_likelihoods_not_custom = len(self.observable_sectors_no_theory_uncertainty)
        indices_mvn_not_custom = jnp.nonzero(
            np.sum(
                selector_matrix_multivariate[:n_likelihoods_not_custom],
                axis=0
            )
        )[0]

        return (
            constraint_dict,
            constraint_no_corr,
            selector_matrix_univariate,
            selector_matrix_multivariate,
            indices_mvn_not_custom,
        )

    def _get_constraints_correlated(self):
        '''
        Returns the constraints and selector matrices for correlated observable sectors.

        Returns
        -------
        constraints_correlated_par_indep_cov : list
            A list containing the multivariate normal distribution constraints with parameter-independent covariance matrices.
        constraints_correlated_par_dep_cov : list
            A list containing the constraints for correlated observable sectors with parameter-dependent covariance matrices.
        selector_matrix : list[jnp.array]
            A list of selector matrices for each correlated observable sector, with shape `(n_likelihoods, n_distributions)`.
        '''

        # constraints for correlated observable sectors with parameter dependent covariance matrix

        n_correlated_likelihoods = len(self._observables_per_likelihood_correlated)
        unique_indices_list = []
        selector_matrix = []
        for i, observables_correlated in enumerate(self.observables_correlated):
            unique_observable_indices = []
            mvn_to_likelihood_map = defaultdict(list)  # maps indices of observables in the set of correlated sectors (MVNs) to likelihoods
            for j, observables_in_likelihood in enumerate(self._observables_per_likelihood_correlated.values()):
                if (
                    j == i  # this is the set of correlated sectors selected in the i loop
                    or j >= len(self.observables_correlated)  # these are the custom likelihoods
                ):
                    obs_indices = tuple(
                        observables_correlated.index(observable)
                        for observable in observables_in_likelihood
                        if (
                            observable in observables_correlated  # a custom likelihood might contain no observable from this set of correlated sectors
                            and observable in self.observables_constrained  # only consider observables that are constrained
                        )
                    )
                    if obs_indices:
                        if obs_indices not in unique_observable_indices:
                            unique_observable_indices.append(
                                obs_indices
                            )
                        mvn_to_likelihood_map[obs_indices].append(j)

            # build selector matrix of (n_correlated_likelihoods, n_mvns)
            sel_matrix = np.zeros((n_correlated_likelihoods, len(unique_observable_indices)))
            for col, indices in enumerate(unique_observable_indices):
                rows = mvn_to_likelihood_map.get(indices, [])
                sel_matrix[rows, col] = 1  # set the entry to 1 if the likelihood depends on this MVN based on the mvn_to_likelihood_map

            unique_indices_list.append([jnp.array(indices, dtype=int) for indices in unique_observable_indices])
            selector_matrix.append(sel_matrix)

        constraints_correlated_par_dep_cov = [
            self.cov_coeff_th_scaled,
            self.std_sm_exp,
            unique_indices_list,
            self.exp_central_scaled,
            self.cov_exp_scaled,
        ]

        # constraints for correlated observable sectors with parameter independent covariance matrix

        mean = []
        standard_deviation = []
        inverse_correlation = []
        for i, unique_indices in enumerate(unique_indices_list):
            mean.append([])
            standard_deviation.append([])
            inverse_correlation.append([])
            cov_exp_scaled = self.cov_exp_scaled[i]
            cov_coeff_th_scaled = self.cov_coeff_th_scaled[i]
            par_monomials = []
            for name in self.observable_sectors_correlated[i]:
                sector = ObservableSector.get(name)
                par_monomial = np.zeros(len(sector.keys_coeff_observable))
                par_monomial[0] = 1.0
                par_monomials.append(par_monomial)
            cov_obs_th_scaled = cov_coeff_to_cov_obs(par_monomials, cov_coeff_th_scaled)
            corr = cov_obs_th_scaled + cov_exp_scaled  # actually correlation matrix as it is parameter independent and rescaled with its own diagonal
            std_sm_exp = self.std_sm_exp[i]
            for index_array in unique_indices:
                index_list = list(index_array)
                mean[i].append(
                    jnp.asarray(
                        np.take(
                            self.exp_central_scaled[i]*std_sm_exp,
                            index_list
                        ),
                        dtype=jnp.float64
                    )
                )
                std = np.take(
                    std_sm_exp,
                    index_list
                )
                standard_deviation[i].append(
                    jnp.asarray(
                        std,
                        dtype=jnp.float64
                    )
                )
                c = np.take(
                    np.take(corr, index_list, axis=0),
                    index_list,
                    axis=1
                )
                inverse_correlation[i].append(
                    jnp.asarray(
                        np.linalg.inv(c),
                        dtype=jnp.float64
                    )
                )

        constraints_correlated_par_indep_cov = [
            unique_indices_list,
            mean,
            standard_deviation,
            inverse_correlation,
        ]

        return constraints_correlated_par_indep_cov, constraints_correlated_par_dep_cov, selector_matrix

    def get_negative_log_likelihood(
            self,
            par_list: List[Tuple[str, str]],
            likelihood: Union[str, Tuple[str, ...]],
            par_dep_cov: bool,
        ):
        '''
        Get a function that computes the negative log-likelihood for a given list of parameters and likelihood, and the corresponding likelihood data

        Parameters
        ----------
        par_list : List[Tuple[str, str]]
            List of tuples specifying the parameters to include in the likelihood evaluation. Each entry is a tuple where the first element is the parameter name and the second element is `R` for real parameters or `I` for imaginary parameters.
        likelihood : Union[str, Tuple[str, ...]]
            The likelihood to evaluate. This can be a string specifying a single likelihood (e.g., 'global' for the combined likelihood, or the name of a specific likelihood), or a tuple of strings specifying a correlated set of likelihoods.
        par_dep_cov : bool
            Whether to use the parameter-dependent covariance matrix for correlated likelihoods.

        Returns
        -------
        negative_log_likelihood : Callable
            A function that computes the negative log-likelihood given an array of parameter values, a scale, and the likelihood data.
        log_likelihood_data : List
            A list containing the data needed for the likelihood evaluation.

        Examples
        --------
        Get the negative log-likelihood function and data for a specific set of parameters and the global likelihood:
        >>> negative_log_likelihood, log_likelihood_data = global_likelihood.get_negative_log_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', par_dep_cov=False
        >>> par_array = jnp.array([1e-8, 1e-8])
        >>> scale = 1000.0
        >>> nll_value = negative_log_likelihood(par_array, scale, log_likelihood_data)

        '''
        # prepare selector matrices for included likelihoods
        if likelihood == 'global':  # for global likelihood, select all non-custom likelihoods
            selector_matrix_no_th_unc_univariate  = self.selector_matrix_no_th_unc_univariate[:len(self.observable_sectors_no_theory_uncertainty)]
            selector_matrix_no_th_unc_multivariate = self.selector_matrix_no_th_unc_multivariate[:len(self.observable_sectors_no_theory_uncertainty)]
            selector_matrix_correlated = [selector_matrix[:len(self.observable_sectors_correlated)] for selector_matrix in self.selector_matrix_correlated]
        else:  # for a specific likelihood, select just the corresponding rows in selector matrices
            if likelihood in self._observables_per_likelihood_no_theory_uncertainty:
                n = list(self._observables_per_likelihood_no_theory_uncertainty).index(likelihood)
                selector_matrix_no_th_unc_univariate = self.selector_matrix_no_th_unc_univariate[[n], :]
                selector_matrix_no_th_unc_multivariate = self.selector_matrix_no_th_unc_multivariate[[n], :]
            else:
                selector_matrix_no_th_unc_univariate = None
                selector_matrix_no_th_unc_multivariate = None
            if likelihood in self._observables_per_likelihood_correlated:
                n = list(self._observables_per_likelihood_correlated).index(likelihood)
                selector_matrix_correlated = [selector_matrix[[n], :] for selector_matrix in self.selector_matrix_correlated]
            else:
                selector_matrix_correlated = [None for _ in self.selector_matrix_correlated]

        log_likelihood_data = [
            self.prediction_data_no_theory_uncertainty,
            self.prediction_data_correlated,
            self.constraints_no_theory_uncertainty,
            self.constraints_correlated_par_indep_cov,
            self.constraints_correlated_par_dep_cov,
            selector_matrix_no_th_unc_univariate,
            selector_matrix_no_th_unc_multivariate,
            selector_matrix_correlated,
        ]

        n_parameters = len(self.parameter_basis_split_re_im)
        par_indices = jnp.array([self.parameter_basis_split_re_im[par] for par in par_list])

        def negative_log_likelihood(
            par_array: jnp.array,
            scale: Union[float, int, jnp.array],
            log_likelihood_data: List,
        ) -> float:

            (
                prediction_data_no_theory_uncertainty,
                prediction_data_correlated,
                constraints_no_theory_uncertainty,
                constraints_correlated_par_indep_cov,
                constraints_correlated_par_dep_cov,
                selector_matrix_no_th_unc_univariate,
                selector_matrix_no_th_unc_multivariate,
                selector_matrix_correlated,
            ) = log_likelihood_data

            par_array_full = jnp.zeros(n_parameters)
            par_array_full = par_array_full.at[par_indices].set(par_array)

            # no theory uncertainty likelihoods
            log_likelihood_no_th_unc_summed = 0.0
            if selector_matrix_no_th_unc_univariate is not None:
                prediction_no_theory_uncertainty = self.prediction_function_no_theory_uncertainty(
                    par_array_full, scale, prediction_data_no_theory_uncertainty
                )
                for distribution_type in constraints_no_theory_uncertainty.keys():
                    if distribution_type == 'MultivariateNormalDistribution':
                        selector_matrix = selector_matrix_no_th_unc_multivariate
                    else:
                        selector_matrix = selector_matrix_no_th_unc_univariate
                    log_likelihood_no_th_unc_summed += jnp.sum(
                        logL_functions_summed[distribution_type](
                            prediction_no_theory_uncertainty,
                            selector_matrix,
                            *constraints_no_theory_uncertainty[distribution_type]
                        )
                    )

            # correlated likelihoods
            prediction_correlated = [
                prediction_function(
                    par_array_full, scale, prediction_data_correlated[i]
                ) for i, prediction_function in enumerate(self.prediction_function_correlated)  # includes predictions and par_monomials
            ]
            n_correlated_sectors = len(selector_matrix_correlated)
            log_likelihood_correlated_summed = 0.0
            if par_dep_cov:
                (cov_coeff_th_scaled,
                 std_sm_exp,
                 observable_indices,
                 exp_central_scaled,
                 cov_exp_scaled,
                ) = constraints_correlated_par_dep_cov
                for i in range(n_correlated_sectors):
                    selector_matrix = selector_matrix_correlated[i]
                    if selector_matrix is not None:
                        predictions, par_monomials = prediction_correlated[i]
                        cov_obs_th_scaled = cov_coeff_to_cov_obs(par_monomials, cov_coeff_th_scaled[i])
                        log_likelihood_correlated_summed += jnp.sum(
                            logL_correlated_sectors_summed(
                                predictions/std_sm_exp[i],
                                selector_matrix,
                                observable_indices[i],
                                exp_central_scaled[i],
                                cov_obs_th_scaled,
                                cov_exp_scaled[i]
                            )
                        )
            else:
                (
                 observable_indices,
                 mean,
                 standard_deviation,
                 inverse_correlation,
                ) = constraints_correlated_par_indep_cov
                logL_function = logL_functions_summed['MultivariateNormalDistribution']
                for i in range(n_correlated_sectors):
                    selector_matrix = selector_matrix_correlated[i]
                    if selector_matrix is not None:
                        predictions, _ = prediction_correlated[i]
                        log_likelihood_correlated_summed += jnp.sum(
                            logL_function(
                                predictions,
                                selector_matrix,
                                observable_indices[i],
                                mean[i],
                                standard_deviation[i],
                                inverse_correlation[i],
                            )
                        )
            return - (log_likelihood_no_th_unc_summed + log_likelihood_correlated_summed)

        return negative_log_likelihood, log_likelihood_data

    def _get_log_likelihood_point_function(self):
        '''
        Returns a JIT-compiled function to compute the information needed for `GlobalLikelihoodPoint` instances.

        Returns
        -------
        log_likelihood_point : Callable
            A function that computes the predictions and log-likelihood contributions for a given parameter array, scale, and likelihood data.
        '''

        n_likelihoods = len(self.likelihoods)

        def log_likelihood_point(
            par_array: jnp.array,
            scale: Union[float, int, jnp.array],
            par_dep_cov: bool,
            prediction_data_no_theory_uncertainty: jnp.array,
            prediction_data_correlated: jnp.array,
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            constraints_correlated_par_indep_cov: Union[List[jnp.array],List[List[jnp.array]]],
            constraints_correlated_par_dep_cov: Union[List[jnp.array],List[List[jnp.array]]],
            selector_matrix_no_th_unc_univariate: jnp.array,
            selector_matrix_no_th_unc_multivariate: jnp.array,
            selector_matrix_correlated: List[jnp.array],
            likelihood_indices_no_theory_uncertainty: jnp.array,
            likelihood_indices_correlated: jnp.array,
            likelihood_indices_global: jnp.array,
        ) -> Tuple[jnp.array]:

            # no theory uncertainty likelihoods and predictions
            prediction_no_theory_uncertainty = self.prediction_function_no_theory_uncertainty(
                par_array, scale, prediction_data_no_theory_uncertainty
            )
            log_likelihood_no_th_unc_univariate = jnp.zeros(len(prediction_no_theory_uncertainty))
            log_likelihood_no_th_unc_multivariate = jnp.zeros((1, len(prediction_no_theory_uncertainty)))
            for distribution_type in constraints_no_theory_uncertainty.keys():
                if distribution_type == 'MultivariateNormalDistribution':
                    log_likelihood_no_th_unc_multivariate = logL_functions[distribution_type](
                        prediction_no_theory_uncertainty,
                        *constraints_no_theory_uncertainty[distribution_type]
                    )
                else:
                    log_likelihood_no_th_unc_univariate += logL_functions[distribution_type](
                        prediction_no_theory_uncertainty,
                        *constraints_no_theory_uncertainty[distribution_type]
                    )

            log_likelihood_no_theory_uncertainty_summed = (
                selector_matrix_no_th_unc_univariate @ log_likelihood_no_th_unc_univariate
                + selector_matrix_no_th_unc_multivariate @ jnp.sum(log_likelihood_no_th_unc_multivariate, axis=1)
            )

            # correlated likelihoods and predictions
            prediction_correlated = [
                prediction_function(
                    par_array, scale, prediction_data_correlated[i]
                ) for i, prediction_function in enumerate(self.prediction_function_correlated)  # includes predictions and par_monomials
            ]
            n_correlated_sectors = len(prediction_correlated)
            log_likelihood_correlated = []
            std_th_exp_correlated_scaled = []
            if par_dep_cov:
                (cov_coeff_th_scaled,
                 std_sm_exp,
                 observable_indices,
                 exp_central_scaled,
                 cov_exp_scaled,
                ) = constraints_correlated_par_dep_cov
                for i in range(n_correlated_sectors):
                    predictions, par_monomials = prediction_correlated[i]
                    cov_obs_th_scaled = cov_coeff_to_cov_obs(par_monomials, cov_coeff_th_scaled[i])
                    std_th_exp_correlated_scaled.append(jnp.sqrt(jnp.diag(cov_obs_th_scaled) + jnp.diag(cov_exp_scaled[i])))
                    log_likelihood_correlated.append(
                        logL_correlated_sectors(
                            predictions/std_sm_exp[i],
                            observable_indices[i],
                            exp_central_scaled[i],
                            cov_obs_th_scaled,
                            cov_exp_scaled[i]
                        )
                    )
            else:
                (
                 observable_indices,
                 mean,
                 standard_deviation,
                 inverse_correlation,
                ) = constraints_correlated_par_indep_cov
                logL_function = logL_functions['MultivariateNormalDistribution']
                for i in range(n_correlated_sectors):
                    predictions, _ = prediction_correlated[i]
                    std_th_exp_correlated_scaled.append(jnp.ones_like(predictions))
                    log_likelihood_correlated.append(
                        logL_function(
                            predictions,
                            observable_indices[i],
                            mean[i],
                            standard_deviation[i],
                            inverse_correlation[i],
                        )
                    )

            n_correlated_likelihoods = len(likelihood_indices_correlated)
            log_likelihood_correlated_summed = jnp.zeros(n_correlated_likelihoods)
            for i in range(n_correlated_sectors):
                logL = jnp.sum(log_likelihood_correlated[i], axis=1)
                logL = jnp.where(jnp.isnan(logL), len(log_likelihood_correlated[i])*LOG_ZERO, logL)
                log_likelihood_correlated_summed += selector_matrix_correlated[i] @ logL

            log_likelihood_summed = jnp.zeros(n_likelihoods)
            log_likelihood_summed = log_likelihood_summed.at[likelihood_indices_no_theory_uncertainty].add(log_likelihood_no_theory_uncertainty_summed)
            log_likelihood_summed = log_likelihood_summed.at[likelihood_indices_correlated].add(log_likelihood_correlated_summed)
            log_likelihood_global = jnp.sum(log_likelihood_summed[likelihood_indices_global])
            log_likelihood_summed = log_likelihood_summed.at[-1].set(log_likelihood_global)
            return (
                prediction_no_theory_uncertainty,
                prediction_correlated,
                log_likelihood_no_th_unc_univariate,
                log_likelihood_no_th_unc_multivariate,
                log_likelihood_correlated,
                log_likelihood_summed,
                std_th_exp_correlated_scaled,
            )
        return jit(log_likelihood_point, static_argnames=["par_dep_cov"])

    def _get_obstable_function(self):
        '''
        Returns a JIT-compiled function to compute the observable table information.

        Returns
        -------
        obstable : Callable
            A function that computes the log-likelihood contributions and related information for a given set of predictions and constraints.
        '''

        @jit
        def obstable(
            prediction_no_theory_uncertainty: jnp.array,
            prediction_correlated: List[jnp.array],
            log_likelihood_no_th_unc_multivariate: jnp.array,
            log_likelihood_correlated: List[jnp.array],
            std_th_exp_correlated_scaled: List[jnp.array],
            constraints_no_theory_uncertainty_no_corr: List[jnp.array],
            indices_mvn_not_custom: jnp.array,
            exp_central_scaled: List[jnp.array],
            std_sm_exp: List[jnp.array],
        ) -> Tuple[jnp.array]:

            # no theory uncertainty sectors
            # including correlations
            log_likelihood_no_th_unc_multivariate = jnp.sum(
                jnp.take(
                    log_likelihood_no_th_unc_multivariate,
                    indices_mvn_not_custom,
                    axis=0
                ),
                axis=0
            )

            # neglecting correlations
            if constraints_no_theory_uncertainty_no_corr is not None:
                log_likelihood_no_th_unc_multivariate_no_corr = logL_functions['NormalDistribution'](
                    prediction_no_theory_uncertainty,
                    *constraints_no_theory_uncertainty_no_corr,
                )
            else:
                log_likelihood_no_th_unc_multivariate_no_corr = jnp.zeros(len(prediction_no_theory_uncertainty))

            # correlated sectors
            # including correlations
            log_likelihood_correlated = [log_likelihood[0] for log_likelihood in log_likelihood_correlated]

            # neglecting correlations
            log_likelihood_correlated_no_corr = []
            exp_central_correlated = []
            std_th_exp_correlated = []
            n_correlated_sectors = len(prediction_correlated)
            for i in range(n_correlated_sectors):
                std_th_exp = std_th_exp_correlated_scaled[i] * std_sm_exp[i]
                exp_central = exp_central_scaled[i] * std_sm_exp[i]
                observable_indices = jnp.arange(len(prediction_correlated[i][0]))
                log_likelihood_correlated_no_corr.append(
                    logL_functions['NormalDistribution'](
                        prediction_correlated[i][0],
                        observable_indices,
                        exp_central,
                        std_th_exp
                    )
                )
                exp_central_correlated.append(exp_central)
                std_th_exp_correlated.append(std_th_exp)

            return (
                log_likelihood_no_th_unc_multivariate,
                log_likelihood_no_th_unc_multivariate_no_corr,
                log_likelihood_correlated,
                log_likelihood_correlated_no_corr,
                exp_central_correlated,
                std_th_exp_correlated,
            )
        return obstable

    def _get_parameter_basis(self):
        '''
        Determines the parameter basis and splits parameters into real and imaginary parts.

        Returns
        -------
        parameter_basis_split_re_im : Dict[Union[str, Tuple[str, str]], int]
            A dictionary mapping parameter names (or tuples of parameter name and 'R'/'I') to their indices in the basis with real and imaginary parts split.
        parameter_basis : Dict[str, int]
            A dictionary mapping parameter names to their indices in the basis without splitting real and imaginary parts.
        '''
        if self.basis_mode == 'rgevolve':
            parameter_basis_split_re_im = get_wc_basis(eft=self.eft, basis=self.basis, sector=None, split_re_im=True)
            parameter_basis = get_wc_basis(eft=self.eft, basis=self.basis, sector=None, split_re_im=False)
        elif self.basis_mode == 'wcxf':
            parameter_basis_split_re_im = get_wc_basis_from_wcxf(eft=self.eft, basis=self.basis, sector=None, split_re_im=True)
            parameter_basis = get_wc_basis_from_wcxf(eft=self.eft, basis=self.basis, sector=None, split_re_im=False)
        else:
            custom_basis = CustomBasis.get(
                ObservableSector.get(self.observable_sectors[0]).custom_basis
            )
            parameter_basis_split_re_im = custom_basis.get_parameter_basis(split_re_im=True)
            parameter_basis = custom_basis.get_parameter_basis(split_re_im=False)
        parameter_basis_split_re_im = {par: i for i, par in enumerate(parameter_basis_split_re_im)}
        parameter_basis = {par: i for i, par in enumerate(parameter_basis)}
        return parameter_basis_split_re_im, parameter_basis

    def _get_par_array(self, par_dict):
        '''
        Converts a parameter dictionary into a JAX array.

        Parameters
        ----------
        par_dict : dict
            A dictionary mapping parameter names (or tuples of parameter name and 'R'/'I') to their values.

        Returns
        -------
        jnp.array
            A JAX array containing the parameter values in the order defined by `parameter_basis_split_re_im`.
        '''
        if not par_dict:
            return jnp.zeros(len(self.parameter_basis_split_re_im))
        elif isinstance(list(par_dict.keys())[0], tuple):
            par_array = np.zeros(len(self.parameter_basis_split_re_im))
            for name, value in par_dict.items():
                if name not in self.parameter_basis_split_re_im:
                    raise ValueError(f"Parameter {name} not found in the parameter basis.")
                par_array[self.parameter_basis_split_re_im[name]] = value
            return jnp.array(par_array)
        else:
            par_array = np.zeros(len(self.parameter_basis_split_re_im))
            for name, value in par_dict.items():
                if (name,'R') not in self.parameter_basis_split_re_im:
                    raise ValueError(f"Parameter {name} not found in the parameter basis.")
                par_array[self.parameter_basis_split_re_im[(name, 'R')]] = value.real
                if (name, 'I') in self.parameter_basis_split_re_im:
                    par_array[self.parameter_basis_split_re_im[(name, 'I')]] = value.imag
            return jnp.array(par_array)

    def parameter_point(self, *args, par_dep_cov: bool = False):
        """
        Create a `GlobalLikelihoodPoint` instance.

        Parameters
        ----------
        *args : tuple
            Positional arguments. The method dispatches
            based on the number and types of these arguments. Accepted input signatures:

              1. `parameter_point(par_dict: dict, scale: Union[float, int], *, par_dep_cov: bool = False)`
                - Create a `GlobalLikelihoodPoint` from a dictionary of parameters and a scale.

              2. `parameter_point(w: wilson.Wilson, *, par_dep_cov: bool = False)`
                - Create a `GlobalLikelihoodPoint` from a `wilson.Wilson` object.

              3. `parameter_point(wc: wilson.wcxf.WC, *, par_dep_cov: bool = False)`
                - Create a `GlobalLikelihoodPoint` from a `wilson.wcxf.WC` object.

              4. `parameter_point(filename: str, *, par_dep_cov: bool = False)`
                - Create a `GlobalLikelihoodPoint` from the path to a WCxf file.

        par_dep_cov : bool, optional
            If `True`, use the parameter dependent covariance matrix for the likelihood point.
            Default is `False`.

        Returns
        -------
        GlobalLikelihoodPoint
            An instance of GlobalLikelihoodPoint with the specified parameters.
        """

        if len(args) == 2:
            par_dict, scale = args
            if not isinstance(par_dict, dict) or not isinstance(scale, (float, int)):
                raise ValueError(
                    "Invalid types of the two positional arguments. Expected a dictionary and scale."
                )
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Wilson):
                par_dict = arg.wc.dict
                scale = arg.wc.scale
            elif isinstance(arg, wcxf.WC):
                par_dict = arg.dict
                scale = arg.scale
            elif isinstance(arg, str):
                with open(arg, 'r') as f:
                    wc = wcxf.WC.load(f)
                par_dict = wc.dict
                scale = wc.scale
            else:
                raise ValueError(
                    "Invalid type of the positional argument. Expected a Wilson or wcxf.WC object, or a filename."
                )
        else:
            raise ValueError("Invalid number of positional arguments. Expected either two (a dictionary and scale) or one (a Wilson or wcxf.WC object, or a filename).")
        return GlobalLikelihoodPoint(self, self._get_par_array(par_dict), scale, par_dep_cov=par_dep_cov)

    def get_compiled_likelihood(
        self,
        par_list: List[Tuple[str, str]],
        likelihood: Union[str, Tuple[str, ...]],
        par_dep_cov: bool = False,
    ):
        '''
        Returns an instance of `CompiledLikelihood` for the specified parameters and likelihood.

        Parameters
        ----------
        par_list : List[Tuple[str, str]]
            List of tuples specifying the parameters to include in the likelihood evaluation. Each entry is a tuple where the first element is the parameter name and the second element is `R` for real parameters or `I` for imaginary parameters.
        likelihood : Union[str, Tuple[str, ...]]
            The likelihood to evaluate. This can be a string specifying a single likelihood (e.g., 'global' for the combined likelihood, or the name of a specific likelihood), or a tuple of strings specifying a correlated set of likelihoods.
        par_dep_cov : bool, optional
            Whether to use the parameter-dependent covariance matrix for correlated likelihoods. Default is `False`.

        Returns
        -------
        CompiledLikelihood
            An instance of `CompiledLikelihood` containing jitted functions for likelihood evaluation.

        Examples
        --------
        Get a `CompiledLikelihood` instance for a specific set of parameters and the global likelihood:
        >>> compiled_likelihood = global_likelihood.get_compiled_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', par_dep_cov=False)
        '''
        if (tuple(par_list), likelihood, par_dep_cov) not in self._cache_compiled_likelihood:
            compiled_likelihood = CompiledLikelihood(
                self,
                par_list,
                likelihood,
                par_dep_cov,
            )
            self._cache_compiled_likelihood[(tuple(par_list), likelihood, par_dep_cov)] = compiled_likelihood
        return self._cache_compiled_likelihood[(tuple(par_list), likelihood, par_dep_cov)]

    def _get_reference_scale(self):
        '''
        Determines the reference scale for the likelihood.

        Returns
        -------
        float
            The reference scale for the likelihood.
        '''
        if self.basis_mode == 'rgevolve':
            return float(reference_scale[self.eft])
        else:
            return ObservableSector.get(self.observable_sectors[0]).scale

    def plot_data_2d(self, par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20, par_dep_cov=False):
        '''
        Computes a grid of chi-squared values over a 2D parameter space for plotting. Returns a dictionary containing the parameter grid and the corresponding chi-squared values.

        Parameters
        ----------
        par_fct : Callable
            A function that takes two arguments (x, y) and returns a dictionary of parameters.
        scale : Union[float, int, Callable]
            The scale at which to evaluate the parameters. This can be a fixed float or int, or a callable that takes (x, y) and returns a scale.
        x_min : float
            The minimum value of the x-axis parameter (in log10 if x_log is `True`).
        x_max : float
            The maximum value of the x-axis parameter (in log10 if x_log is `True`).
        y_min : float
            The minimum value of the y-axis parameter (in log10 if y_log is `True`).
        y_max : float
            The maximum value of the y-axis parameter (in log10 if y_log is `True`).
        x_log : bool, optional
            Whether to use a logarithmic scale for the x-axis. Default is `False`.
        y_log : bool, optional
            Whether to use a logarithmic scale for the y-axis. Default is `False`.
        steps : int, optional
            The number of steps in each dimension for the grid. Default is `20`.
        par_dep_cov : bool, optional
            Whether to use the parameter-dependent covariance matrix for correlated likelihoods. Default is `False`.

        Returns
        -------
        plotdata : Dict
            A dictionary containing the parameter grid and the corresponding chi-squared values for each likelihood. The keys are the names of the likelihoods, and the values are dictionaries with keys `x`, `y`, and `z`, where `x` and `y` are the parameter grids and `z` is the chi-squared grid.

        Examples
        --------
        Define a function that maps (x, y) to a dictionary of parameters:
        >>> def par_func(x, y):
        ...     return {'lq1_1111': x, 'lq3_1111': y}

        Obtain the 2D chi-squared grid for two parameters over specified ranges:

        >>> plot_data = gl.plot_data_2d(par_func, scale=1000.0, x_min=-1e-8, x_max=1e-8, y_min=-1e-8, y_max=1e-8, steps=50)
        '''
        if x_log:
            _x = jnp.logspace(x_min, x_max, steps)
        else:
            _x = jnp.linspace(x_min, x_max, steps)
        if y_log:
            _y = jnp.logspace(y_min, y_max, steps)
        else:
            _y = jnp.linspace(y_min, y_max, steps)
        x, y = jnp.meshgrid(_x, _y)
        xy = jnp.array([x, y]).reshape(2, steps**2).T
        xy_enumerated = list(enumerate(xy))
        if isinstance(scale, Number):
            scale_fct = partial(_scale_fct_fixed, scale=scale)
        else:
            scale_fct = scale
        ll = partial(_log_likelihood_2d, gl=self, par_fct=par_fct, scale_fct=scale_fct, par_dep_cov=par_dep_cov)
        ll_dict_list_enumerated = map(ll, xy_enumerated)  # no multiprocessing for now
        ll_dict_list = [
            ll_dict[1] for ll_dict in
            sorted(ll_dict_list_enumerated, key=itemgetter(0))
        ]
        plotdata = {}
        keys = ll_dict_list[0].keys()  # look at first dict to fix keys
        for k in keys:
            z = -2 * np.array([ll_dict[k] for ll_dict in ll_dict_list]).reshape((steps, steps))
            plotdata[k] = {'x': x, 'y': y, 'z': z}
        return plotdata

def _scale_fct_fixed(*args, scale=0):
    """
    This is a helper function that is necessary because multiprocessing requires
    a picklable (i.e. top-level) object for parallel computation.
    """
    return scale

def _log_likelihood_2d(xy_enumerated, gl, par_fct, scale_fct, par_dep_cov=False):
    """Compute the likelihood on a 2D grid of 2 Wilson coefficients.

    This function is necessary because multiprocessing requires a picklable
    (i.e. top-level) object for parallel computation.
    """
    number, (x, y) = xy_enumerated
    pp = gl.parameter_point(par_fct(x, y), scale_fct(x, y), par_dep_cov=par_dep_cov)
    ll_dict = pp.log_likelihood_dict()
    return (number, ll_dict)
