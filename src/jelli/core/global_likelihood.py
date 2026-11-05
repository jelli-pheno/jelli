from typing import List, Dict, Union, Tuple
from itertools import chain
from functools import partial
import threading
import warnings
import weakref
from jax import jit, numpy as jnp
import numpy as np
from numbers import Number, Integral
from operator import itemgetter
from wilson import Wilson, wcxf
import networkx as nx
from rgevolve.tools import get_wc_basis, reference_scale
from .observable_sector import ObservableSector
from .measurement import Measurement
from .custom_basis import CustomBasis
from .global_likelihood_point import GlobalLikelihoodPoint
from .theory_correlations import TheoryCorrelations, UncorrelatedCovarianceSource, cut_of_full_block
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
from ..utils.par_helpers import get_wc_basis_from_wcxf, get_active_monomial_indices
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
    constraints_correlated_par_dep_cov : list
        The constraints for observables in correlated sectors with parameter-dependent covariance.
        Its first item, the theory covariance tensors, is None: each likelihood requests the tensors
        for its monomials (see `_get_cov_th_scaled`).
    cov_th_scaled_par_indep : list[jnp.array]
        For each correlated group, the theory covariance of its constrained observables where all
        parameters vanish, scaled like `cov_exp_scaled`.
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
    _cov_th_sources : list[list[list[TheoryCovarianceSource or UncorrelatedCovarianceSource or None]]]
        For each correlated group, the sources of the theory covariance blocks (lower triangle), from
        which the blocks are built on demand. None for a sector pair without theory correlations; an
        `UncorrelatedCovarianceSource` for a sector without an entry with itself.
    _cov_th_full : dict
        The full float64 blocks built so far, keyed by (group, row sector, column sector). Once a
        block is built, it is kept, and the source of the block drops its correlations.
    _cov_th_cuts : weakref.WeakValueDictionary
        The blocks restricted to some monomials, keyed by block and monomials. A block stays in it
        while a likelihood uses it, so that likelihoods with the same monomials share it.
    _cov_th_point : dict
        For each `cov_order`, the monomials and the theory covariance tensors of the point function.
    _cov_th_lock : threading.Lock
        Held while a block is looked up, built, cached and its correlations released, so that
        threads that request blocks of the same `GlobalLikelihood` do not interfere.
    _max_kept_fraction : float
        At `cov_order >= 2`, a sector is restricted to the monomials that can be nonzero only if that
        keeps at most this fraction of them (1/sqrt(2)); otherwise all its monomials are used.
    _obstable : callable
        The JIT-compiled function to compute the observable table information.
    _cache_compiled_likelihood : dict
        A cache for `CompiledLikelihood` instances to avoid redundant computations.

    Methods
    -------
    load(path: str) -> None
        Initializes `ObservableSector`, `Measurement`, `TheoryCorrelations`, and `ExperimentalCorrelations` classes by loading data from the specified path.
    get_negative_log_likelihood(par_list: List[Tuple[str, str]], likelihood: Union[str, Tuple[str, ...]], cov_order: int = 0) -> Tuple[Callable, List]
        Returns a function to compute the negative log-likelihood for given parameters and likelihood.
    parameter_point(*args, cov_order: int = 0) -> GlobalLikelihoodPoint
        Returns a `GlobalLikelihoodPoint` instance for the specified parameter values.
    get_compiled_likelihood(par_list: List[Tuple[str, str]], likelihood: Union[str, Tuple[str, ...]], cov_order: int = 0) -> CompiledLikelihood
        Returns an instance of `CompiledLikelihood` for the specified parameters and likelihood.
    plot_data_2d(par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20, cov_order=0) -> Dict
        Computes a grid of chi-squared values over a 2D parameter space for plotting. Returns a dictionary containing the parameter grid and the corresponding chi-squared values.
    _get_observable_sectors(include_observable_sectors, exclude_observable_sectors) -> Tuple[List[str], List[str], str]
        Determines the observable sectors to include in the likelihood based on inclusion/exclusion lists.
    _setup_correlated_sectors() -> None
        Determines correlated observable sector groups and computes constrained-only data for the correlated likelihood.
    _setup_constraints_correlated() -> None
        Computes constraints and selector matrices for correlated observable sectors using constrained-only indices.
    _get_cov_coeff_indices(par_indices, cov_order) -> List[List[Optional[np.ndarray]]]
        Selects the theory covariance monomials that can be nonzero for the parameters at `par_indices`.
    _get_cov_th_scaled(cov_coeff_indices) -> List[Optional[List[List[Optional[jnp.array]]]]]
        Returns the theory covariance blocks for the given monomials, built on demand and shared.
    _log_likelihood_point(par_array, scale, cov_order=0) -> Tuple
        Computes the predictions and log-likelihood contributions at one point (for `GlobalLikelihoodPoint`).
    _get_custom_likelihoods(custom_likelihoods) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Processes custom likelihoods.
    _get_observables_per_likelihood() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Constructs dictionaries mapping likelihood names to lists of observables for both no theory uncertainty and correlated sectors.
    _get_prediction_function_no_theory_uncertainty() -> Callable
        Returns a prediction function for observables with no theory uncertainty.
    _get_constraints_no_theory_uncertainty(observables, observable_lists_per_likelihood=None) -> Tuple[Dict, Dict, jnp.array, jnp.array, jnp.array]
        Returns the constraints and selector matrices for observables with no theory uncertainty.
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

    >>> negative_log_likelihood, log_likelihood_data = gl.get_negative_log_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', cov_order=0)

    Get an instance of `CompiledLikelihood` for specific parameters and a likelihood:

    >>> compiled_likelihood = gl.get_compiled_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', cov_order=0)

    Access the parameter basis:

    >>> parameter_basis = gl.parameter_basis
    >>> parameter_basis_split_re_im = gl.parameter_basis_split_re_im

    Access the basis mode:

    >>> basis_mode = gl.basis_mode

    Access the observables included in the likelihood:

    >>> observables_gaussian = gl.observables_gaussian
    >>> observables_no_theory_uncertainty = gl.observables_no_theory_uncertainty

    '''

    # At cov_order >= 2 a sector is cut to the monomials that can be nonzero only if the cut keeps at
    # most this fraction of them; otherwise all its monomials are used (WP5). With 1/sqrt(2), the copy
    # of a block for a cut, together with the int16 correlations kept for it, takes less memory than
    # the full block: at most 3/4 of it if both sectors are cut, and 0.96 if one is.
    _max_kept_fraction = 2 ** -0.5

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

        self._setup_correlated_sectors()

        # observables of Gaussian sectors that are not in any correlated group
        # (see `_setup_correlated_sectors`) are appended at the end
        sectors_correlated = set(chain.from_iterable(self.observable_sectors_correlated))
        self.observables_gaussian = list(chain.from_iterable(
            self.observables_correlated
            )) + list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_gaussian
            if observable_sector not in sectors_correlated
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

        self._setup_constraints_correlated()

        self._log_likelihood_point_function = self._get_log_likelihood_point_function()
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
            cov_order=0,
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

    def _setup_correlated_sectors(self):
        '''
        Determines correlated observable sector groups and computes constrained-only
        data for the correlated likelihood.

        Only observables that are constrained by an included measurement are included in
        the calculation. A group in which no sector has a constrained observable is
        therefore not set up. A warning lists the Gaussian sectors that have no
        constrained observable.

        Sets the following attributes on self:

        Always set:
            observable_sectors_correlated : list[list[str]]
            observables_correlated : list[list[str]]
                Full observable names per group (for custom likelihood classification).
            observables_correlated_constrained : list[list[str]]
                Constrained-only observable names per group.
            cov_exp_scaled : list[jnp.array]
                Experimental covariance matrices (constrained-only, 2D).
            exp_central_scaled : list[jnp.array]
            std_sm_exp : list[jnp.array]
            std_exp : list[jnp.array]
            prediction_data_correlated : list[list[list[jnp.array]]]
            prediction_function_correlated : list[callable]
            _cov_th_sources : list[list[list[TheoryCovarianceSource or UncorrelatedCovarianceSource or None]]]
                The sources of the 4D theory covariance blocks (constrained-only,
                lower triangle), from which `_get_cov_th_scaled` builds the blocks
                that a likelihood needs. A source is None if the correlation files
                have no entry for the two sectors, which the POPxf format defines
                as uncorrelated. A sector without an entry with itself has
                uncorrelated coefficient uncertainties: its source is an
                `UncorrelatedCovarianceSource`, whose block is
                T[m, m, α, α] = σ̂_mα², zero elsewhere.
            _cov_th_full, _cov_th_cuts, _cov_th_point
                The blocks built so far (see `_get_cov_th_scaled`), all empty here.
        '''

        # build connectivity graph (metadata only, no large arrays loaded): an edge for each pair
        # of sectors that has an entry in the correlation files or a nonzero experimental
        # correlation, so that the groups are the parts that are correlated

        G = nx.Graph()
        G.add_nodes_from(self.observable_sectors_gaussian)
        corr_exp = {}  # (i, j) -> corr_array for lower triangle (i >= j)
        for i, row_sector in enumerate(self.observable_sectors_gaussian):
            obs_row = ObservableSector.get(row_sector).observable_names
            for j, col_sector in enumerate(self.observable_sectors_gaussian[:i+1]):
                obs_col = ObservableSector.get(col_sector).observable_names
                has_th = TheoryCorrelations.has_correlations(obs_row, obs_col)
                exp_corr = ExperimentalCorrelations.get_data('correlations', self.include_measurements, obs_row, obs_col)
                if exp_corr is not None:
                    corr_exp[(i, j)] = exp_corr
                if has_th or exp_corr is not None:
                    G.add_edge(row_sector, col_sector)

        components = list(nx.connected_components(G))
        components = [sorted(list(group)) for group in components]
        components = sorted(components, key=lambda c: self.observable_sectors_gaussian.index(c[0]))

        # Gaussian sectors without any constrained observable; a group consisting only of
        # such sectors is not set up
        unconstrained = [
            name for name in self.observable_sectors_gaussian
            if not any(obs in self.observables_constrained for obs in ObservableSector.get(name).observable_names)
        ]
        if unconstrained:
            warnings.warn(
                f"The observable sector(s) {unconstrained} have no observable constrained by the included "
                f"measurements and are not included in the likelihood. Exclude them with "
                f"`exclude_observable_sectors` to avoid this warning.",
                UserWarning,
                stacklevel=3,
            )
        components = [group for group in components if not set(group) <= set(unconstrained)]
        self.observable_sectors_correlated = components


        # process each connected component: compute constrained-only data,
        # covariance matrices, and prediction functions

        self.observables_correlated = []
        self.observables_correlated_constrained = []
        self.std_sm_exp = []
        self.exp_central_scaled = []
        self.std_exp = []
        self.cov_exp_scaled = []
        self.prediction_data_correlated = []
        self.prediction_function_correlated = []
        self._cov_th_sources = []
        self._cov_th_full = {}
        self._cov_th_cuts = weakref.WeakValueDictionary()
        self._cov_th_point = {}
        self._cov_th_lock = threading.Lock()

        for group in components:
            # map from group-local index to gaussian-list index
            g_idx = [self.observable_sectors_gaussian.index(name) for name in group]

            # --- per-sector: observable names, constrained indices, scaled uncertainties ---

            obs_full = []
            obs_constrained_group = []
            std_th_scaled_group = []
            std_exp_scaled_group = []
            std_sm_exp_group = []
            exp_central_scaled_group = []
            std_exp_group = []
            constrained_idx_group = []
            prediction_functions = []
            prediction_data = []

            for s, sector_name in enumerate(group):
                sector = ObservableSector.get(sector_name)
                obs_all = sector.observable_names
                obs_c = [obs for obs in obs_all if obs in self.observables_constrained]
                c_idx = [obs_all.index(obs) for obs in obs_c]
                obs_full.extend(obs_all)
                obs_constrained_group.extend(obs_c)
                constrained_idx_group.append(c_idx)

                std_exp = np.asarray(ExperimentalCorrelations.get_data('uncertainties', self.include_measurements, obs_all))
                exp_central = np.asarray(ExperimentalCorrelations.get_data('central', self.include_measurements, obs_all))
                std_th = sector.observable_uncertainties
                std_sm = sector.observable_uncertainties_SM

                std_exp_c = std_exp[c_idx]
                exp_central_c = exp_central[c_idx]
                std_th_c = std_th[:, c_idx]
                std_sm_c = std_sm[c_idx]

                _std_sm_exp = std_exp_c * np.sqrt(1 + (std_sm_c / std_exp_c)**2)
                std_th_scaled_group.append(std_th_c / _std_sm_exp)
                std_exp_scaled_group.append(std_exp_c / _std_sm_exp)
                std_sm_exp_group.append(_std_sm_exp)
                exp_central_scaled_group.append(exp_central_c / _std_sm_exp)
                std_exp_group.append(std_exp_c)

                # prediction function for the constrained observables (the list may be empty)
                fn, data = sector.get_prediction_function(
                    self.eft, self.basis,
                    observables=obs_c
                )
                prediction_functions.append(fn)
                prediction_data.append(data)

            self.observables_correlated.append(obs_full)
            self.observables_correlated_constrained.append(obs_constrained_group)
            self.std_sm_exp.append(jnp.array(np.concatenate(std_sm_exp_group)))
            self.exp_central_scaled.append(jnp.array(np.concatenate(exp_central_scaled_group)))
            self.std_exp.append(jnp.array(np.concatenate(std_exp_group)))
            self.prediction_data_correlated.append(prediction_data)

            def _make_group_prediction(pred_fns):
                def prediction(par_array, scale, prediction_data):
                    polynomial_predictions = [jnp.empty(0)]
                    par_monomials = []
                    for pred_fn, data in zip(pred_fns, prediction_data):
                        polynomial_prediction, par_monomial = pred_fn(par_array, scale, data)
                        polynomial_predictions.append(polynomial_prediction)
                        par_monomials.append(par_monomial)
                    polynomial_predictions = jnp.concatenate(polynomial_predictions, axis=-1)
                    return polynomial_predictions, par_monomials
                return prediction
            self.prediction_function_correlated.append(_make_group_prediction(prediction_functions))

            # --- per-block: covariance matrices (lower triangle) ---

            sub_exp = []
            sub_th = []

            for i in range(len(group)):
                row_exp = []
                row_th = []
                c_idx_row = constrained_idx_group[i]
                obs_i = ObservableSector.get(group[i]).observable_names
                for j in range(i + 1):
                    c_idx_col = constrained_idx_group[j]
                    obs_j = ObservableSector.get(group[j]).observable_names
                    gi, gj = g_idx[i], g_idx[j]

                    # experimental covariance block (constrained-only)
                    if gi >= gj:
                        exp_block = corr_exp.get((gi, gj))
                        flipped = False
                    else:
                        exp_block = corr_exp.get((gj, gi))
                        flipped = True

                    if exp_block is not None:
                        exp_corr_np = np.asarray(exp_block)
                        if flipped:
                            exp_corr_np = exp_corr_np.T
                        exp_corr_sub = exp_corr_np[np.ix_(c_idx_row, c_idx_col)]
                        row_exp.append(exp_corr_sub * np.outer(std_exp_scaled_group[i], std_exp_scaled_group[j]))
                    else:
                        row_exp.append(np.zeros((len(c_idx_row), len(c_idx_col))))

                    # source of the theory covariance block, read from HDF5 and restricted to the
                    # constrained observables; the block itself is built when a likelihood needs it
                    source = TheoryCorrelations.get_covariance_source(
                        obs_i, obs_j, c_idx_row, c_idx_col,
                        std_th_scaled_group[i], std_th_scaled_group[j]
                    )
                    if source is None and i == j:
                        # no entry of the sector with itself: by the POPxf format, its coefficient
                        # uncertainties are uncorrelated
                        source = UncorrelatedCovarianceSource(std_th_scaled_group[i])
                    # None marks a sector pair without theory correlations
                    row_th.append(source)

                sub_exp.append(row_exp)
                sub_th.append(row_th)

            # assemble full experimental covariance matrix from lower triangle blocks
            n_sectors = len(sub_exp)
            cov_exp_blocks = [[None]*n_sectors for _ in range(n_sectors)]
            for i in range(n_sectors):
                for j in range(n_sectors):
                    if i >= j:
                        cov_exp_blocks[i][j] = sub_exp[i][j]
                    else:
                        shape = sub_exp[j][i].shape
                        cov_exp_blocks[i][j] = np.zeros((shape[1], shape[0]))
            cov_exp_tril = np.tril(np.block(cov_exp_blocks))
            cov_exp_full = cov_exp_tril + cov_exp_tril.T - np.diag(np.diag(cov_exp_tril))
            self.cov_exp_scaled.append(jnp.array(cov_exp_full))
            self._cov_th_sources.append(sub_th)

        # precompute the indices of the monomials of order <= 1 per sector, for cov_order=1: its
        # theory covariance is that of the polynomials truncated after their linear terms
        # (only used for the covariance contraction — predictions always use all coefficients).
        # A coefficient (w1, w2, c) has order (0 if w1=='' else 1) + (0 if w2=='' else 1).
        self._coeff_indices_linear = []
        for group in components:
            group_indices = []
            for sector_name in group:
                sector = ObservableSector.get(sector_name)
                indices = [k for k, (w1, w2, _) in enumerate(sector.keys_coeff_observable)
                           if (0 if w1 == '' else 1) + (0 if w2 == '' else 1) <= 1]
                group_indices.append(np.array(indices, dtype=int))
            self._coeff_indices_linear.append(group_indices)

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
            tuple(observable_sectors): self.observables_correlated_constrained[i]
            for i, observable_sectors in enumerate(self.observable_sectors_correlated)
            }

        return observables_per_likelihood_no_theory_uncertainty, observables_per_likelihood_correlated

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

    def _setup_constraints_correlated(self):
        '''
        Computes constraints and selector matrices for correlated observable sectors
        using constrained-only indices.

        Sets the following attributes on self:
            constraints_correlated_par_indep_cov : list
                Constraints with parameter-independent covariance matrices.
            constraints_correlated_par_dep_cov : list
                Constraints for parameter-dependent covariance matrices. The first item, the
                theory covariance tensors, is None: every likelihood requests its own.
            cov_th_scaled_par_indep : list[jnp.array]
                The scaled theory covariance of each group where all parameters vanish.
            selector_matrix_correlated : list[jnp.array]
                Selector matrices for each correlated group.
        '''

        # build unique observable indices and selector matrices
        # using observables_correlated_constrained for index mapping

        n_correlated_likelihoods = len(self._observables_per_likelihood_correlated)
        unique_indices_list = []
        selector_matrix = []
        for i, obs_constrained in enumerate(self.observables_correlated_constrained):
            unique_observable_indices = []
            mvn_to_likelihood_map = defaultdict(list)
            for j, observables_in_likelihood in enumerate(self._observables_per_likelihood_correlated.values()):
                if (
                    j == i  # this is the set of correlated sectors selected in the i loop
                    or j >= len(self.observables_correlated)  # these are the custom likelihoods
                ):
                    obs_indices = tuple(
                        obs_constrained.index(observable)
                        for observable in observables_in_likelihood
                        if observable in obs_constrained
                    )
                    if obs_indices:
                        if obs_indices not in unique_observable_indices:
                            unique_observable_indices.append(obs_indices)
                        mvn_to_likelihood_map[obs_indices].append(j)

            sel_matrix = np.zeros((n_correlated_likelihoods, len(unique_observable_indices)))
            for col, indices in enumerate(unique_observable_indices):
                rows = mvn_to_likelihood_map.get(indices, [])
                sel_matrix[rows, col] = 1

            unique_indices_list.append([jnp.array(indices, dtype=int) for indices in unique_observable_indices])
            selector_matrix.append(sel_matrix)

        # par_dep_cov constraints; the theory covariance tensors are requested by each likelihood
        self.constraints_correlated_par_dep_cov = [
            None,
            self.std_sm_exp,
            unique_indices_list,
            self.exp_central_scaled,
            self.cov_exp_scaled,
        ]

        # par_indep_cov constraints: the 2D theory covariance where all parameters vanish. Only the
        # constant monomial, the first of every sector, contributes to it, so only the blocks of that
        # monomial are built; they are freed at the end of this method.
        constant = [[np.zeros(1, dtype=int)] * len(group) for group in self.observable_sectors_correlated]
        cov_th_constant = self._get_cov_th_scaled(constant)
        self.cov_th_scaled_par_indep = []

        mean = []
        standard_deviation = []
        inverse_correlation = []
        for i, unique_indices in enumerate(unique_indices_list):
            mean.append([])
            standard_deviation.append([])
            inverse_correlation.append([])
            cov_exp_scaled = self.cov_exp_scaled[i]

            cov_obs_th_scaled = cov_coeff_to_cov_obs(
                [np.ones(1)] * len(self.observable_sectors_correlated[i]), cov_th_constant[i]
            )
            self.cov_th_scaled_par_indep.append(cov_obs_th_scaled)

            corr = cov_obs_th_scaled + cov_exp_scaled
            std_sm_exp = self.std_sm_exp[i]
            for index_array in unique_indices:
                index_list = list(index_array)
                mean[i].append(
                    jnp.asarray(
                        np.take(self.exp_central_scaled[i] * std_sm_exp, index_list),
                        dtype=jnp.float64
                    )
                )
                std = np.take(std_sm_exp, index_list)
                standard_deviation[i].append(
                    jnp.asarray(std, dtype=jnp.float64)
                )
                c = np.take(np.take(corr, index_list, axis=0), index_list, axis=1)
                inverse_correlation[i].append(
                    jnp.asarray(np.linalg.inv(c), dtype=jnp.float64)
                )

        self.constraints_correlated_par_indep_cov = [
            unique_indices_list,
            mean,
            standard_deviation,
            inverse_correlation,
        ]
        self.selector_matrix_correlated = selector_matrix

    @staticmethod
    def _check_cov_order(cov_order):
        '''
        Check that `cov_order` is a non-negative integer.

        Bools are rejected: since `True == 1`, `cov_order=True` would otherwise silently select
        `cov_order=1`, where only the constant and the linear terms enter the theory covariance.

        Parameters
        ----------
        cov_order : int
            Order up to which the monomials of the parameters enter the theory covariance
            (see `get_negative_log_likelihood`).

        Returns
        -------
        int
            `cov_order` as an `int`.
        '''
        if isinstance(cov_order, bool):
            raise TypeError(
                f"`cov_order` must be an int, not the bool {cov_order}: use `cov_order=2` for a fully "
                f"parameter-dependent theory covariance and `cov_order=0` for a parameter-independent one."
            )
        if not isinstance(cov_order, Integral):
            raise TypeError(f"`cov_order` must be an int (0, 1 or 2), not {cov_order!r}.")
        if cov_order < 0:
            raise ValueError(f"`cov_order` must be 0, 1 or 2, not {cov_order}.")
        return int(cov_order)

    def _get_cov_coeff_indices(self, par_indices, cov_order):
        '''
        Selects the monomials of the theory covariance that can be nonzero when only the
        parameters at `par_indices` are nonzero, and only those of order <= 1 if
        `cov_order == 1`. The others vanish identically, so leaving them out of the
        covariance contraction does not change it.

        At `cov_order >= 2`, a sector whose selection keeps more than the fraction
        `_max_kept_fraction` of its monomials keeps all of them instead: contracting the
        monomials that vanish changes nothing, and the full block is shared by every likelihood
        that needs it, while a copy of most of it would only save a little time. At
        `cov_order == 1` the selection is always used, since the monomials of order 2 must be
        left out there.

        Parameters
        ----------
        par_indices : np.ndarray
            Indices in `parameter_basis_split_re_im` of the parameters that can be nonzero.
        cov_order : int
            Expansion order of the theory covariance, ``1`` or ``>= 2``.

        Returns
        -------
        cov_coeff_indices : list[list[np.ndarray or None]]
            For each correlated group and each of its sectors, the indices of the monomials
            to keep, or None if all of them are kept.
        '''
        cov_coeff_indices = []
        for i, group in enumerate(self.observable_sectors_correlated):
            group_indices = []
            for s, sector_name in enumerate(group):
                sector = ObservableSector.get(sector_name)
                # the evolution matrices as used in the predictions, and the columns they act on
                sector_indices, evolution_matrices = self.prediction_data_correlated[i][s][:2]
                indices = get_active_monomial_indices(
                    sector.keys_pars, sector.keys_coeff_observable,
                    sector_indices, evolution_matrices, par_indices,
                )
                if cov_order == 1:
                    indices = np.intersect1d(indices, self._coeff_indices_linear[i][s])
                n_monomials = len(sector.keys_coeff_observable)
                keep_all = len(indices) == n_monomials or (
                    cov_order >= 2 and len(indices) > self._max_kept_fraction * n_monomials
                )
                group_indices.append(None if keep_all else indices)
            cov_coeff_indices.append(group_indices)
        return cov_coeff_indices

    def _get_cov_th_scaled(self, cov_coeff_indices):
        '''
        Returns the theory covariance blocks of the correlated groups for the given monomials.

        A full block (all monomials of both sectors) is built from its source the first time it
        is needed and then kept; its source drops its correlations, so that a block never holds
        both. A block for some of the monomials is built from the source, or taken from the
        full block if that exists, and is kept in a weak cache: likelihoods that need the same
        monomials share it, and it is freed with the last likelihood that uses it.

        Parameters
        ----------
        cov_coeff_indices : list
            For each correlated group, None if its blocks are not needed, or a list with, for each
            sector, the indices of its monomials, or None for all of them (as returned by
            `_get_cov_coeff_indices`).

        Returns
        -------
        list[list[list[jnp.array or None]] or None]
            For each needed group, the blocks of its lower triangle, with None for a sector pair
            without theory correlations; None for a group that is not needed.
        '''
        tensors = []
        for g, (sources, indices) in enumerate(zip(self._cov_th_sources, cov_coeff_indices)):
            if indices is None:
                tensors.append(None)
                continue
            tensors.append([
                [self._get_cov_th_block(g, i, j, indices[i], indices[j]) for j in range(i + 1)]
                for i in range(len(sources))
            ])
        return tensors

    def __getstate__(self):
        # a lock can be neither copied nor pickled: `copy.deepcopy` and pickling leave it out, and
        # `__setstate__` gives the copy a lock of its own
        state = self.__dict__.copy()
        state.pop('_cov_th_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cov_th_lock = threading.Lock()

    def _get_cov_th_block(self, group, i, j, row_monomials, col_monomials):
        '''
        One block of `_get_cov_th_scaled`.

        The lookup, the build, the caching and the release of the correlations are one step under
        `_cov_th_lock`: otherwise a thread could build a cut from correlations that another thread
        has just released for the full block, or two threads could build the same block. One lock
        for all blocks also builds one block at a time, which keeps the memory peak low.
        '''
        source = self._cov_th_sources[group][i][j]
        if source is None:  # no theory correlations between the two sectors
            return None
        with self._cov_th_lock:
            full = self._cov_th_full.get((group, i, j))
            if row_monomials is None and col_monomials is None:
                if full is None:
                    full = source.build()
                    source.release()
                    self._cov_th_full[(group, i, j)] = full
                return full
            key = (group, i, j, _monomials_key(row_monomials), _monomials_key(col_monomials))
            block = self._cov_th_cuts.get(key)
            if block is None:
                if full is not None:
                    block = cut_of_full_block(full, row_monomials, col_monomials)
                else:
                    block = source.build(row_monomials, col_monomials)
                self._cov_th_cuts[key] = block
            return block

    def get_negative_log_likelihood(
            self,
            par_list: List[Tuple[str, str]],
            likelihood: Union[str, Tuple[str, ...]],
            cov_order: int = 0,
        ):
        '''
        Get a function that computes the negative log-likelihood for a given
        list of parameters and likelihood, and the corresponding likelihood data.

        Parameters
        ----------
        par_list : List[Tuple[str, str]]
            List of tuples specifying the parameters to include in the
            likelihood evaluation. Each entry is a tuple where the first
            element is the parameter name and the second element is ``'R'``
            for real parameters or ``'I'`` for imaginary parameters.
        likelihood : Union[str, Tuple[str, ...]]
            The likelihood to evaluate. This can be a string specifying a
            single likelihood (e.g., ``'global'`` for the combined likelihood,
            or the name of a specific likelihood), or a tuple of strings
            specifying a correlated set of likelihoods.
        cov_order : int, optional
            Order up to which the monomials of the parameters enter the theory
            covariance. The covariance of two observables is
            Σ_αα′ V_α Σ̃_αα′ V_α′, a sum over pairs of monomials V_α, V_α′ of the
            parameters, where Σ̃ is the covariance of the polynomial coefficients
            (POPxf paper, "Observable Uncertainties and Correlations").
            ``cov_order=n`` keeps the terms in which both monomials have order at
            most ``n``, so it is the covariance of the polynomials truncated after
            order ``n``. The monomial of a coefficient ``(w1, w2, c)`` has order
            ``(0 if w1=='' else 1) + (0 if w2=='' else 1)``.

            - ``0``: the constant terms only, which gives the covariance where all
              parameters vanish (precomputed, the cheapest).
            - ``1``: the constant and the linear terms.
            - ``2`` (or more): all terms, the full parameter dependence.

            Each is the covariance of the same correlated coefficient uncertainties,
            so it is positive semi-definite at every point. Only the covariance is
            affected; the predictions always use all polynomial coefficients. At
            ``cov_order >= 1``, the covariance contraction skips the monomials that
            vanish identically when the parameters not in ``par_list`` are zero,
            taking RG mixing into account, so the result is the same. Default is
            ``0``. A bool raises a ``TypeError``.

        Returns
        -------
        negative_log_likelihood : Callable
            A function that computes the negative log-likelihood given an
            array of parameter values, a scale, and the likelihood data.
        log_likelihood_data : List
            A list containing the data needed for the likelihood evaluation.
            At ``cov_order >= 1`` it holds the theory covariance tensors for the
            monomials of ``par_list``. They are built when this method is called,
            unless a likelihood with the same monomials still holds them; then
            both share them. A tensor is freed when the last likelihood data that
            holds it is deleted. At ``cov_order == 0`` it holds no such tensors.

        Examples
        --------
        >>> negative_log_likelihood, log_likelihood_data = gl.get_negative_log_likelihood(
        ...     par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')],
        ...     likelihood='global', cov_order=0)
        >>> nll_value = negative_log_likelihood(jnp.array([1e-8, 1e-8]), 1000.0, log_likelihood_data)
        '''
        cov_order = self._check_cov_order(cov_order)

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
        # a group whose selector is all zeros, one that the selected likelihood does not use, gets
        # None too: it is not evaluated, and its theory covariance tensors are not built
        selector_matrix_correlated = [
            None if selector_matrix is None or not np.any(selector_matrix) else selector_matrix
            for selector_matrix in selector_matrix_correlated
        ]

        n_parameters = len(self.parameter_basis_split_re_im)
        par_indices = jnp.array([self.parameter_basis_split_re_im[par] for par in par_list])

        # the theory covariance tensors: the monomials that can be nonzero for the parameters in
        # par_list, and at cov_order == 1 only those of order <= 1, the theory covariance of the
        # polynomials truncated after their linear terms; None at cov_order == 0, where the
        # covariance where all parameters vanish is used instead
        if cov_order >= 1:
            cov_coeff_indices = [
                # a group that no selected likelihood uses is never evaluated: no tensors
                None if selector_matrix is None else indices
                for indices, selector_matrix in zip(
                    self._get_cov_coeff_indices(np.asarray(par_indices), cov_order),
                    selector_matrix_correlated,
                )
            ]
            cov_th_scaled = self._get_cov_th_scaled(cov_coeff_indices)
        else:
            cov_coeff_indices = None
            cov_th_scaled = None
        constraints_correlated_par_dep_cov = [cov_th_scaled, *self.constraints_correlated_par_dep_cov[1:]]

        log_likelihood_data = [
            self.prediction_data_no_theory_uncertainty,
            self.prediction_data_correlated,
            self.constraints_no_theory_uncertainty,
            self.constraints_correlated_par_indep_cov,
            constraints_correlated_par_dep_cov,
            selector_matrix_no_th_unc_univariate,
            selector_matrix_no_th_unc_multivariate,
            selector_matrix_correlated,
        ]

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
            if cov_order >= 1:
                (cov_th_scaled,
                 std_sm_exp,
                 observable_indices,
                 exp_central_scaled,
                 cov_exp_scaled,
                ) = constraints_correlated_par_dep_cov
                for i in range(n_correlated_sectors):
                    selector_matrix = selector_matrix_correlated[i]
                    if selector_matrix is not None:
                        predictions, par_monomials = prediction_correlated[i]
                        par_monomials = [m if idx is None else jnp.take(m, idx, axis=-1)
                                         for m, idx in zip(par_monomials, cov_coeff_indices[i])]
                        cov_obs_th_scaled = cov_coeff_to_cov_obs(par_monomials, cov_th_scaled[i])
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

    def _log_likelihood_point(self, par_array, scale, cov_order=0):
        '''
        Computes the predictions and the log-likelihood contributions at one point, the information
        that `GlobalLikelihoodPoint` needs.

        At `cov_order >= 1` the theory covariance uses the monomials of order <= 1 at `cov_order == 1`
        and all monomials at `cov_order >= 2`, for any point. Their tensors are built at the first
        call with that `cov_order` and then kept, so that later points do not build them again.

        Parameters
        ----------
        par_array : jnp.array
            Values of all parameters in `parameter_basis_split_re_im`.
        scale : float or int or jnp.array
            The scale of the parameters.
        cov_order : int, optional
            Expansion order of the theory covariance. Default is ``0``.

        Returns
        -------
        tuple
            The output of `_log_likelihood_point_function`.
        '''
        if cov_order >= 1 and cov_order not in self._cov_th_point:
            if cov_order == 1:
                # the monomials of order <= 1, and None for a sector that has no others, so that its
                # full block is used rather than copied as a cut of all its monomials
                indices = [
                    [
                        None if len(linear) == len(ObservableSector.get(name).keys_coeff_observable) else linear
                        for linear, name in zip(group_linear, sectors)
                    ]
                    for group_linear, sectors in zip(self._coeff_indices_linear, self.observable_sectors_correlated)
                ]
            else:
                indices = [[None] * len(group) for group in self.observable_sectors_correlated]
            self._cov_th_point[cov_order] = (indices, self._get_cov_th_scaled(indices))
        cov_th_scaled = self._cov_th_point[cov_order][1] if cov_order >= 1 else None
        return self._log_likelihood_point_function(
            par_array,
            scale,
            cov_order=cov_order,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            prediction_data_correlated=self.prediction_data_correlated,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty,
            constraints_correlated_par_indep_cov=self.constraints_correlated_par_indep_cov,
            constraints_correlated_par_dep_cov=[cov_th_scaled, *self.constraints_correlated_par_dep_cov[1:]],
            selector_matrix_no_th_unc_univariate=self.selector_matrix_no_th_unc_univariate,
            selector_matrix_no_th_unc_multivariate=self.selector_matrix_no_th_unc_multivariate,
            selector_matrix_correlated=self.selector_matrix_correlated,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_correlated=self._likelihood_indices_correlated,
            likelihood_indices_global=self._likelihood_indices_global,
        )

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
            cov_order: int,
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
            if cov_order >= 1:
                # the monomials of the tensors that `_log_likelihood_point` requested: at cov_order 1
                # those of order <= 1, the theory covariance of the polynomials truncated after their
                # linear terms, and all at cov_order >= 2 (Python values, read when the function is
                # traced for this cov_order)
                cov_coeff_indices = self._cov_th_point[cov_order][0]
                (cov_th_scaled,
                 std_sm_exp,
                 observable_indices,
                 exp_central_scaled,
                 cov_exp_scaled,
                ) = constraints_correlated_par_dep_cov
                for i in range(n_correlated_sectors):
                    predictions, par_monomials = prediction_correlated[i]
                    par_monomials = [m if idx is None else jnp.take(m, idx, axis=-1)
                                     for m, idx in zip(par_monomials, cov_coeff_indices[i])]
                    cov_obs_th_scaled = cov_coeff_to_cov_obs(par_monomials, cov_th_scaled[i])
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
        return jit(log_likelihood_point, static_argnames=["cov_order"])

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

    def parameter_point(self, *args, cov_order: int = 0):
        """
        Create a `GlobalLikelihoodPoint` instance.

        Parameters
        ----------
        *args : tuple
            Positional arguments. The method dispatches
            based on the number and types of these arguments. Accepted input signatures:

              1. `parameter_point(par_dict: dict, scale: Union[float, int], *, cov_order: int = 0)`
                - Create a `GlobalLikelihoodPoint` from a dictionary of parameters and a scale.

              2. `parameter_point(w: wilson.Wilson, *, cov_order: int = 0)`
                - Create a `GlobalLikelihoodPoint` from a `wilson.Wilson` object.

              3. `parameter_point(wc: wilson.wcxf.WC, *, cov_order: int = 0)`
                - Create a `GlobalLikelihoodPoint` from a `wilson.wcxf.WC` object.

              4. `parameter_point(filename: str, *, cov_order: int = 0)`
                - Create a `GlobalLikelihoodPoint` from the path to a WCxf file.

        cov_order : int, optional
            Order up to which the monomials of the parameters enter the theory covariance:
            ``0`` the constant terms only (the covariance where all parameters vanish), ``1``
            the constant and the linear terms, ``2`` all terms; see
            `get_negative_log_likelihood`.
            Default is ``0``. A bool raises a ``TypeError``.

        Returns
        -------
        GlobalLikelihoodPoint
            An instance of GlobalLikelihoodPoint with the specified parameters.
        """
        cov_order = self._check_cov_order(cov_order)

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
        return GlobalLikelihoodPoint(self, self._get_par_array(par_dict), scale, cov_order=cov_order)

    def get_compiled_likelihood(
        self,
        par_list: List[Tuple[str, str]],
        likelihood: Union[str, Tuple[str, ...]],
        cov_order: int = 0,
    ):
        '''
        Returns an instance of `CompiledLikelihood` for the specified parameters and likelihood.

        Parameters
        ----------
        par_list : List[Tuple[str, str]]
            List of tuples specifying the parameters to include in the likelihood evaluation. Each entry is a tuple where the first element is the parameter name and the second element is `R` for real parameters or `I` for imaginary parameters.
        likelihood : Union[str, Tuple[str, ...]]
            The likelihood to evaluate. This can be a string specifying a single likelihood (e.g., 'global' for the combined likelihood, or the name of a specific likelihood), or a tuple of strings specifying a correlated set of likelihoods.
        cov_order : int, optional
            Order up to which the monomials of the parameters enter the theory covariance:
            ``0`` the constant terms only (the covariance where all parameters vanish), ``1``
            the constant and the linear terms, ``2`` all terms; see
            `get_negative_log_likelihood`.
            Default is ``0``. A bool raises a ``TypeError``.

        Returns
        -------
        CompiledLikelihood
            An instance of `CompiledLikelihood` containing jitted functions for likelihood evaluation.

        Examples
        --------
        Get a `CompiledLikelihood` instance for a specific set of parameters and the global likelihood:
        >>> compiled_likelihood = global_likelihood.get_compiled_likelihood(par_list=[('lq1_1111', 'R'), ('lq3_1111', 'R')], likelihood='global', cov_order=0)
        '''
        cov_order = self._check_cov_order(cov_order)
        if (tuple(par_list), likelihood, cov_order) not in self._cache_compiled_likelihood:
            compiled_likelihood = CompiledLikelihood(
                self,
                par_list,
                likelihood,
                cov_order,
            )
            self._cache_compiled_likelihood[(tuple(par_list), likelihood, cov_order)] = compiled_likelihood
        return self._cache_compiled_likelihood[(tuple(par_list), likelihood, cov_order)]

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

    def plot_data_2d(self, par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20, cov_order=0):
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
        cov_order : int, optional
            Order up to which the monomials of the parameters enter the theory covariance:
            ``0`` the constant terms only (the covariance where all parameters vanish), ``1``
            the constant and the linear terms, ``2`` all terms; see
            `get_negative_log_likelihood`.
            Default is ``0``. A bool raises a ``TypeError``.

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
        cov_order = self._check_cov_order(cov_order)
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
        ll = partial(_log_likelihood_2d, gl=self, par_fct=par_fct, scale_fct=scale_fct, cov_order=cov_order)
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

def _monomials_key(monomials):
    '''A hashable key for the monomial indices of one axis of a theory covariance block (None: all).'''
    return None if monomials is None else np.asarray(monomials, dtype=np.int64).tobytes()

def _scale_fct_fixed(*args, scale=0):
    """
    This is a helper function that is necessary because multiprocessing requires
    a picklable (i.e. top-level) object for parallel computation.
    """
    return scale

def _log_likelihood_2d(xy_enumerated, gl, par_fct, scale_fct, cov_order=0):
    """Compute the likelihood on a 2D grid of 2 Wilson coefficients.

    This function is necessary because multiprocessing requires a picklable
    (i.e. top-level) object for parallel computation.
    """
    number, (x, y) = xy_enumerated
    pp = gl.parameter_point(par_fct(x, y), scale_fct(x, y), cov_order=cov_order)
    ll_dict = pp.log_likelihood_dict()
    return (number, ll_dict)
