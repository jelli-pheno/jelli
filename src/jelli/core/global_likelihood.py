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
    coeff_cov_to_obs_cov,
    logL_correlated_sectors,
)
from ..utils.par_helpers import get_wc_basis_from_wcxf
from collections import defaultdict

class GlobalLikelihood():

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
            self.cov_th_scaled,
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
            self.constraints_no_theory_uncertainty_decorrelated,
            self.selector_matrix_univariate,
            self.selector_matrix_multivariate,
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
            selector_matrix_univariate=self.selector_matrix_univariate,
            selector_matrix_multivariate=self.selector_matrix_multivariate,
            selector_matrix_correlated=self.selector_matrix_correlated,
            likelihood_indices_no_theory_uncertainty=self._likelihood_indices_no_theory_uncertainty,
            likelihood_indices_correlated=self._likelihood_indices_correlated,
            likelihood_indices_global=self._likelihood_indices_global,
        )
        (
            self.sm_prediction_no_theory_uncertainty,
            self.sm_prediction_correlated,
            self.sm_log_likelihood_univariate_per_observable,
            self.sm_log_likelihood_multivariate_per_observable,
            self.sm_log_likelihood_correlated_per_observable,
            self.sm_log_likelihood,
            self.sm_standard_deviation_th_correlated,
        ) = self._log_likelihood_point(
            self._get_par_array({}),
            self._reference_scale,
            par_dep_cov=False,
        )

    @classmethod
    def load(cls, path):
        # load all observable sectors
        ObservableSector.load(path)
        # load all measurements
        Measurement.load(path)
        # load all theory correlations
        TheoryCorrelations.load(path)
        # load all experimental correlations
        ExperimentalCorrelations.load()

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

    def _get_observable_sectors_correlated(self):

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

        cov_th_scaled = []
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
            cov_th_scaled.append(sub_th)

            n_sectors = len(sub_exp)
            cov_exp = np.empty((n_sectors, n_sectors), dtype=object).tolist()
            for i in range(n_sectors):
                for j in range(n_sectors):
                    if i >= j:
                        cov_exp[i][j] = sub_exp[i][j]
                    else:
                        shape = sub_exp[j][i].shape
                        cov_exp[i][j] = np.zeros((shape[1], shape[0]))
            cov_exp_matrix_tril = np.tril(np.block(cov_exp))
            sub_exp = cov_exp_matrix_tril + cov_exp_matrix_tril.T - np.diag(np.diag(cov_exp_matrix_tril))
            cov_exp_scaled.append(jnp.array(sub_exp))

        return (
            observable_sectors_correlated,
            cov_th_scaled,
            cov_exp_scaled,
            exp_central_scaled,
            std_sm_exp,
            std_exp_list,
        )

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

    def _get_observables_per_likelihood(self):

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

        # decorrelated MVN constraints
        if 'MultivariateNormalDistribution' in constraints:
            constraint_decorrelated = [
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['observable_indices'])),
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['central_value'])),
                jnp.asarray(np.concatenate(constraints['MultivariateNormalDistribution']['standard_deviation'])),
            ]
        else:
            constraint_decorrelated = []

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
            constraint_decorrelated,
            selector_matrix_univariate,
            selector_matrix_multivariate,
            indices_mvn_not_custom,
        )

    def _get_constraints_correlated(self):

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
            self.cov_th_scaled,
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
            cov_matrix_exp = self.cov_exp_scaled[i]
            cov_matrix_th_scaled = self.cov_th_scaled[i]
            par_monomials = []
            for name in self.observable_sectors_correlated[i]:
                sector = ObservableSector.get(name)
                par_monomial = np.zeros(len(sector.keys_coeff_observable))
                par_monomial[0] = 1.0
                par_monomials.append(par_monomial)
            cov_matrix_th = coeff_cov_to_obs_cov(par_monomials, cov_matrix_th_scaled)
            corr_matrix = cov_matrix_th + cov_matrix_exp  # actually correlation matrix as it is rescaled
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
                corr = np.take(
                    np.take(corr_matrix, index_list, axis=0),
                    index_list,
                    axis=1
                )
                inverse_correlation[i].append(
                    jnp.asarray(
                        np.linalg.inv(corr),
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

    def _get_log_likelihood_point_function(self):
        prediction_function_no_theory_uncertainty = self.prediction_function_no_theory_uncertainty
        prediction_function_correlated = self.prediction_function_correlated
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
            selector_matrix_univariate: jnp.array,
            selector_matrix_multivariate: jnp.array,
            selector_matrix_correlated: List[jnp.array],
            likelihood_indices_no_theory_uncertainty: jnp.array,
            likelihood_indices_correlated: jnp.array,
            likelihood_indices_global: jnp.array,
        ) -> Tuple[jnp.array]:

            # no theory uncertainty likelihoods and predictions
            prediction_no_theory_uncertainty = prediction_function_no_theory_uncertainty(
                par_array, scale, prediction_data_no_theory_uncertainty
            )
            log_likelihood_univariate_per_observable = jnp.zeros(len(prediction_no_theory_uncertainty))
            log_likelihood_multivariate_per_observable = jnp.zeros((1, len(prediction_no_theory_uncertainty)))
            for distribution_type in constraints_no_theory_uncertainty.keys():
                if distribution_type == 'MultivariateNormalDistribution':
                    log_likelihood_multivariate_per_observable = logL_functions[distribution_type](
                        prediction_no_theory_uncertainty,
                        *constraints_no_theory_uncertainty[distribution_type]
                    )
                else:
                    log_likelihood_univariate_per_observable += logL_functions[distribution_type](
                        prediction_no_theory_uncertainty,
                        *constraints_no_theory_uncertainty[distribution_type]
                    )

            log_likelihood_no_theory_uncertainty = (
                selector_matrix_univariate @ log_likelihood_univariate_per_observable
                + selector_matrix_multivariate @ jnp.sum(log_likelihood_multivariate_per_observable, axis=1)
            )

            # correlated likelihoods and predictions
            prediction_correlated = [
                prediction_function(
                    par_array, scale, prediction_data_correlated[i]
                ) for i, prediction_function in enumerate(prediction_function_correlated)  # includes predictions and par_monomials
            ]
            n_correlated_sectors = len(prediction_correlated)
            log_likelihood_correlated_per_observable = []
            standard_deviation_th_correlated = []
            if par_dep_cov:
                (cov_th_scaled,
                 std_sm_exp,
                 observable_indices,
                 exp_central_scaled,
                 cov_matrix_exp) = constraints_correlated_par_dep_cov
                for i in range(n_correlated_sectors):
                    predictions, par_monomials = prediction_correlated[i]
                    cov_matrix_th = coeff_cov_to_obs_cov(par_monomials, cov_th_scaled[i])
                    standard_deviation_th_correlated.append(jnp.sqrt(jnp.diag(cov_matrix_th)))
                    log_likelihood_correlated_per_observable.append(
                        logL_correlated_sectors(
                            predictions/std_sm_exp[i],
                            observable_indices[i],
                            exp_central_scaled[i],
                            cov_matrix_th,
                            cov_matrix_exp[i]
                        )
                    )
            else:
                (
                 observable_indices,
                 mean,
                 standard_deviation,
                 inverse_correlation,
                ) = constraints_correlated_par_indep_cov
                logpdf_function = logL_functions['MultivariateNormalDistribution']
                for i in range(n_correlated_sectors):
                    predictions, _ = prediction_correlated[i]
                    standard_deviation_th_correlated.append(jnp.ones_like(predictions))
                    log_likelihood_correlated_per_observable.append(
                        logpdf_function(
                            predictions,
                            observable_indices[i],
                            mean[i],
                            standard_deviation[i],
                            inverse_correlation[i],
                        )
                    )

            n_correlated_likelihoods = len(likelihood_indices_correlated)
            log_likelihood_correlated = jnp.zeros(n_correlated_likelihoods)
            for i in range(n_correlated_sectors):
                logpdf = jnp.sum(log_likelihood_correlated_per_observable[i], axis=1)
                logpdf = jnp.where(jnp.isnan(logpdf), -len(log_likelihood_correlated_per_observable[i])*100., logpdf)
                log_likelihood_correlated += selector_matrix_correlated[i] @ logpdf

            log_likelihood = jnp.zeros(n_likelihoods)
            log_likelihood = log_likelihood.at[likelihood_indices_no_theory_uncertainty].add(log_likelihood_no_theory_uncertainty)
            log_likelihood = log_likelihood.at[likelihood_indices_correlated].add(log_likelihood_correlated)
            log_likelihood_global = jnp.sum(log_likelihood[likelihood_indices_global])
            log_likelihood = log_likelihood.at[-1].set(log_likelihood_global)
            return (
                prediction_no_theory_uncertainty,
                prediction_correlated,
                log_likelihood_univariate_per_observable,
                log_likelihood_multivariate_per_observable,
                log_likelihood_correlated_per_observable,
                log_likelihood,
                standard_deviation_th_correlated,
            )
        return jit(log_likelihood_point, static_argnames=["par_dep_cov"])

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

    def parameter_point(self, *args, par_dep_cov: bool = False):
        """
        Create a GlobalLikelihoodPoint instance.

        Accepted input signatures:
        --------------------------
        1. parameter_point(par_dict: dict, scale: Union[float, int], *, par_dep_cov: bool = False)
            - Create a GlobalLikelihoodPoint from a dictionary of parameters and a scale.

        2. parameter_point(w: wilson.Wilson, *, par_dep_cov: bool = False)
            - Create a GlobalLikelihoodPoint from a wilson.Wilson object.

        3. parameter_point(wc: wilson.wcxf.WC, *, par_dep_cov: bool = False)
            - Create a GlobalLikelihoodPoint from a wilson.wcxf.WC object.

        4. parameter_point(filename: str, *, par_dep_cov: bool = False)
            - Create a GlobalLikelihoodPoint from the path to a WCxf file.

        Parameters:
        -----------
        *args :
            Positional arguments as described above. The method dispatches
            based on the number and types of these arguments.
        par_dep_cov : bool, optional
            If True, use the parameter dependent covariance matrix for the likelihood point.
            Default is False.

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

    def _get_reference_scale(self):
        if self.basis_mode == 'rgevolve':
            return float(reference_scale[self.eft])
        else:
            return ObservableSector.get(self.observable_sectors[0]).scale

    def plot_data_2d(self, par_fct, scale, x_min, x_max, y_min, y_max, x_log=False, y_log=False, steps=20, par_dep_cov=False):
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
