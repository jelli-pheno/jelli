from typing import List, Dict, Tuple, Any, Callable, Union, Optional
from itertools import chain
from functools import partial
from jax import grad, jit, numpy as jnp
import numpy as np
from .observable_sector import ObservableSector
from .measurements import Measurement
from ..utils.distributions import logpdf_functions
from multipledispatch import dispatch
from rgevolve.tools import get_wc_basis, reference_scale
from ..utils.wc_helpers import get_wc_basis_from_wcxf
from .custom_basis import CustomBasis
from .global_likelihood_point import GlobalLikelihoodPoint
from numbers import Number
from operator import itemgetter
from wilson import Wilson, wcxf

class GlobalLikelihood():

    def __init__(
        self,
        eft=None,
        basis=None,
        custom_basis=None,
        include_observable_sectors=None,
        exclude_observable_sectors=None,
        custom_likelihoods=None,
    ):

        if custom_basis is not None:
            if eft is not None or basis is not None:
                raise ValueError("Please provide either `custom_basis`, or both `eft` and `basis`, but not both.")
        elif eft is not None and basis is None or basis is not None and eft is None:
            raise ValueError("Please provide the `eft` when using the `basis` and vice versa.")
        self.eft = eft
        self.basis = basis
        self.custom_basis = custom_basis

        (
            self.observable_sectors_gaussian,
            self.observable_sectors_no_theory_uncertainty,
            self.basis_mode
        ) = self._get_observable_sectors(
            include_observable_sectors,
            exclude_observable_sectors
        )
        self.observable_sectors = self.observable_sectors_gaussian + self.observable_sectors_no_theory_uncertainty
        self.observables_gaussian = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_gaussian
        ))
        self.observables_no_theory_uncertainty = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        ))

        self.parameter_basis_split_re_im, self.parameter_basis = self._get_parameter_basis()
        self._reference_scale = self._get_reference_scale()

        self.prediction_data_gaussian = [
            ObservableSector.get(observable_sector).get_prediction_data(self.eft, self.basis)
            for observable_sector in self.observable_sectors_gaussian
        ]
        self.prediction_data_no_theory_uncertainty = [
            ObservableSector.get(observable_sector).get_prediction_data(self.eft, self.basis)
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        ]
        self.prediction_function_gaussian = self._get_prediction_function_gaussian()
        self.prediction_function_no_theory_uncertainty = self._get_prediction_function_no_theory_uncertainty()

        self._observables_per_likelihood_gaussian, self._observables_per_likelihood_no_theory_uncertainty = self._get_observables_per_likelihood(custom_likelihoods)
        self._likelihoods_no_theory_uncertainty = list(self._observables_per_likelihood_no_theory_uncertainty.keys())
        self._likelihoods_gaussian = list(self._observables_per_likelihood_gaussian.keys())
        self.likelihoods = sorted(set(self._likelihoods_no_theory_uncertainty) | set(self._likelihoods_gaussian))

        self._likelihood_indices_no_theory_uncertainty = jnp.array([self.likelihoods.index(likelihood) for likelihood in self._likelihoods_no_theory_uncertainty])
        self._likelihood_indices_gaussian = jnp.array([self.likelihoods.index(likelihood) for likelihood in self._likelihoods_gaussian])

        self.constraints_no_theory_uncertainty = self._get_constraints_no_theory_uncertainty(
            self.observables_no_theory_uncertainty,
            list(self._observables_per_likelihood_no_theory_uncertainty.values())
            )

        self.constraints_gaussian = self._get_constraints_gaussian()

        self._log_likelihood_sm = None

        self.log_likelihood_function = self._get_log_likelihood_function()
        self.delta_log_likelihood_function = self._get_delta_log_likelihood_function()
        self.chi2_function = self._get_chi2_function()

        self.log_likelihood = partial(
            self.log_likelihood_function,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            prediction_data_gaussian=self.prediction_data_gaussian,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty,
            constraints_gaussian=self.constraints_gaussian,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_gaussian=self._likelihood_indices_gaussian,
        )
        self.delta_log_likelihood = partial(
            self.delta_log_likelihood_function,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            prediction_data_gaussian=self.prediction_data_gaussian,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty,
            constraints_gaussian=self.constraints_gaussian,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_gaussian=self._likelihood_indices_gaussian,
            log_likelihood_sm=self.log_likelihood_sm
        )
        self.chi2 = partial(
            self.chi2_function,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            prediction_data_gaussian=self.prediction_data_gaussian,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty,
            constraints_gaussian=self.constraints_gaussian,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_gaussian=self._likelihood_indices_gaussian,
        )

    @classmethod
    def load(cls, path):
        # load all observable sectors
        ObservableSector.load(path)
        # load all measurements
        Measurement.load(path)

    def _get_observable_sectors(self, include_observable_sectors, exclude_observable_sectors):
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

    def _get_custom_likelihoods(self, custom_likelihoods):
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

    def _get_observables_per_likelihood(self, custom_likelihoods):

        custom_likelihoods_gaussian, custom_likelihoods_no_theory_uncertainty = self._get_custom_likelihoods(custom_likelihoods)

        observables_per_likelihood_gaussian = {
            observable_sector: ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_gaussian
        }
        observables_per_likelihood_gaussian.update({
            'global': self.observables_gaussian
        })
        observables_per_likelihood_gaussian.update(custom_likelihoods_gaussian)

        observables_per_likelihood_no_theory_uncertainty = {
            observable_sector: ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        }
        observables_per_likelihood_no_theory_uncertainty.update({
            'global': self.observables_no_theory_uncertainty
        })
        observables_per_likelihood_no_theory_uncertainty.update(custom_likelihoods_no_theory_uncertainty)

        return observables_per_likelihood_gaussian, observables_per_likelihood_no_theory_uncertainty

    def _get_prediction_function_gaussian(self):

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in self.observable_sectors_gaussian
        ]

        @jit
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

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in self.observable_sectors_no_theory_uncertainty
        ]

        @jit
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

        constraint_dict = {}

        constraints = Measurement.get_constraints(observables, distribution_types=[
            'NumericalDistribution',
            'NormalDistribution',
            'HalfNormalDistribution',
            'GammaDistributionPositive',
        ])

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

        if observable_lists_per_likelihood is not None:
            # selector matrix for univariate distributions
            selector_matrix_univariate = jnp.array([
                np.isin(observables, likelihood_observables).astype(int)
                for likelihood_observables in observable_lists_per_likelihood
            ])
            for distribution in constraint_dict:
                constraint_dict[distribution].insert(0, selector_matrix_univariate)

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
                [jnp.asarray(unique_mvnd_blocks[k]['logpdf_normalization_per_observable']) for k in all_mvnd_keys],
            ]
            if observable_lists_per_likelihood is not None:
                # Create selector matrix (n_likelihoods x n_contributions)
                selector_matrix_multivariate = np.zeros((n_likelihoods, n_contributions))
                for i, mvnd_keys in enumerate(mvnd_keys_per_likelihood):
                    for key in mvnd_keys:
                        selector_matrix_multivariate[i, mvnd_key_to_index[key]] = 1.0
                selector_matrix_multivariate = jnp.array(selector_matrix_multivariate)
                constraint_dict['MultivariateNormalDistribution'].insert(0, selector_matrix_multivariate)
        return constraint_dict

    def _get_constraints_gaussian(self):  # TODO: dummy function, returns empty dict
        return {}

    def _get_log_likelihood_function(self):

        prediction_function_no_theory_uncertainty = self.prediction_function_no_theory_uncertainty
        prediction_function_gaussian = self.prediction_function_gaussian
        n_likelihoods = len(self.likelihoods)

        @jit
        def log_likelihood(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data_no_theory_uncertainty: jnp.array,
            prediction_data_gaussian: jnp.array,
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            constraints_gaussian: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            likelihood_indices_no_theory_uncertainty: jnp.array,
            likelihood_indices_gaussian: jnp.array,
        ) -> jnp.array:
            predictions_no_theory_uncertainty = prediction_function_no_theory_uncertainty(
                par_array, scale, prediction_data_no_theory_uncertainty
            )
            log_likelihood_no_theory_uncertainty = jnp.zeros(len(likelihood_indices_no_theory_uncertainty))
            for distribution_type in constraints_no_theory_uncertainty.keys():
                log_likelihood_no_theory_uncertainty += logpdf_functions[distribution_type](
                    predictions_no_theory_uncertainty,
                    *constraints_no_theory_uncertainty[distribution_type]
                )
            predictions_gaussian, par_monomials = prediction_function_gaussian(
                par_array, scale, prediction_data_gaussian
            )
            log_likelihood_gaussian = jnp.zeros(len(likelihood_indices_gaussian))
            # TODO: compute logpdf for gaussian likelihoods here

            log_likelihood = jnp.zeros(n_likelihoods)
            log_likelihood = log_likelihood.at[likelihood_indices_no_theory_uncertainty].add(log_likelihood_no_theory_uncertainty)
            log_likelihood = log_likelihood.at[likelihood_indices_gaussian].add(log_likelihood_gaussian)
            return log_likelihood
        return log_likelihood

    def _get_delta_log_likelihood_function(self):

        log_likelihood_function = self.log_likelihood_function

        @jit
        def delta_log_likelihood(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data_no_theory_uncertainty: jnp.array,
            prediction_data_gaussian: jnp.array,
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            constraints_gaussian: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            likelihood_indices_no_theory_uncertainty: jnp.array,
            likelihood_indices_gaussian: jnp.array,
            log_likelihood_sm: jnp.array,
        ) -> jnp.array:
            return log_likelihood_function(
                par_array, scale,
                prediction_data_no_theory_uncertainty,
                prediction_data_gaussian,
                constraints_no_theory_uncertainty,
                constraints_gaussian,
                likelihood_indices_no_theory_uncertainty,
                likelihood_indices_gaussian,
            ) - log_likelihood_sm
        return delta_log_likelihood

    def _get_chi2_function(self):

        log_likelihood_function = self.log_likelihood_function

        @jit
        def chi2(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data_no_theory_uncertainty: jnp.array,
            prediction_data_gaussian: jnp.array,
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            constraints_gaussian: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
            likelihood_indices_no_theory_uncertainty: jnp.array,
            likelihood_indices_gaussian: jnp.array,
        ) -> jnp.array:
            return -2 * log_likelihood_function(
                par_array, scale,
                prediction_data_no_theory_uncertainty,
                prediction_data_gaussian,
                constraints_no_theory_uncertainty,
                constraints_gaussian,
                likelihood_indices_no_theory_uncertainty,
                likelihood_indices_gaussian,
            )
        return chi2

    def _get_parameter_basis(self):
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

    @dispatch(dict, (int, float))
    def parameter_point(self, par_dict, scale):
        par_array = self._get_par_array(par_dict)
        return GlobalLikelihoodPoint(self, par_array, scale)

    @dispatch(wcxf.WC)
    def parameter_point(self, wc):
        if wc.eft != self.eft:
            raise ValueError(f"Wilson coefficients are defined in the {wc.eft} but the likelihood is defined in the {self.eft}.")
        if wc.basis != self.basis:
            raise ValueError(f"Wilson coefficients are defined in the {wc.basis} basis but the likelihood is defined in the {self.basis} basis.")
        return self.parameter_point(wc.dict, wc.scale)

    @dispatch(Wilson)
    def parameter_point(self, w):
        return self.parameter_point(w.wc)

    @dispatch(str)
    def parameter_point(self, filename):
        with open(filename, 'r') as f:
            wc = wcxf.WC.load(f)
        return self.parameter_point(wc)

    @property
    def log_likelihood_sm(self):
        if self._log_likelihood_sm is None:
            self._log_likelihood_sm = self.log_likelihood(
                self._get_par_array({}), self._reference_scale,
            )
        return self._log_likelihood_sm

    def _get_reference_scale(self):
        if self.basis_mode == 'rgevolve':
            return reference_scale[self.eft]
        else:
            return ObservableSector.get(self.observable_sectors[0]).scale

    def _delta_log_likelihood_dict(self, par_array, scale):
        return dict(zip(
            self.likelihoods,
            self.delta_log_likelihood(par_array, scale)
        ))

    def _chi2_dict(self, par_array, scale):
        return dict(zip(
            self.likelihoods,
            self.chi2(par_array, scale)
        ))

    def plot_data_2d(self, par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20):
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
        ll = partial(_log_likelihood_2d, gl=self, par_fct=par_fct, scale_fct=scale_fct)
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

def _log_likelihood_2d(xy_enumerated, gl, par_fct, scale_fct):
    """Compute the likelihood on a 2D grid of 2 Wilson coefficients.

    This function is necessary because multiprocessing requires a picklable
    (i.e. top-level) object for parallel computation.
    """
    number, (x, y) = xy_enumerated
    pp = gl.parameter_point(par_fct(x, y), scale_fct(x, y))
    ll_dict = pp.log_likelihood_dict()
    return (number, ll_dict)
