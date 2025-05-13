from typing import List, Dict, Tuple, Any, Callable, Union, Optional
from itertools import chain
from functools import partial
from jax import grad, jit, numpy as jnp
from .observable_sector import ObservableSector
from .measurements import Measurement
from ..utils.distributions import logpdf_functions


class GlobalLikelihood():

    _default_bases = {'SMEFT': 'Warsaw', 'WET': 'flavio'}

    def __init__(
        self,
        eft='SMEFT',
        basis=None,
        include_observable_sectors=None,
        exclude_observable_sectors=None,
        custom_likelihoods=None,
    ):
        self.eft = eft
        self.basis = basis or self._default_bases[eft]
        (
            self.observable_sectors_gaussian,
            self.observable_sectors_no_theory_uncertainty
        ) = self._get_observable_sectors(
            include_observable_sectors,
            exclude_observable_sectors
        )
        self.observables_gaussian = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_gaussian
        ))
        self.observables_no_theory_uncertainty = list(chain.from_iterable(
            ObservableSector.get(observable_sector).observable_names
            for observable_sector in self.observable_sectors_no_theory_uncertainty
        ))

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

        self.constraints_no_theory_uncertainty = self._get_constraints_no_theory_uncertainty()
        self.logpdf_function_no_theory_uncertainty = self._get_logpdf_function_no_theory_uncertainty()
        self.global_logpdf_function_no_theory_uncertainty = self._get_global_logpdf_function_no_theory_uncertainty()

        self.global_logpdf_no_theory_uncertainty = partial(
            self.global_logpdf_function_no_theory_uncertainty,
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty
        )
        self.grad_global_logpdf_no_theory_uncertainty = partial(
            jit(grad(
                self.global_logpdf_function_no_theory_uncertainty,
                argnums=(0, 1)
            )),
            prediction_data_no_theory_uncertainty=self.prediction_data_no_theory_uncertainty,
            constraints_no_theory_uncertainty=self.constraints_no_theory_uncertainty
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
        available_observable_sectors = set(ObservableSector.get_all_names(self.eft))
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
        observable_sectors_gaussian = []
        observable_sectors_no_theory_uncertainty = []
        for observable_sector in observable_sectors:
            if ObservableSector.get(observable_sector).observable_uncertainties is None:
                observable_sectors_no_theory_uncertainty.append(observable_sector)
            else:
                observable_sectors_gaussian.append(observable_sector)
        return observable_sectors_gaussian, observable_sectors_no_theory_uncertainty

    def _get_prediction_function_gaussian(self):

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in self.observable_sectors_gaussian
        ]

        @jit
        def prediction(
            wc_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data: List[List[jnp.array]]
        ) -> jnp.array:
            polynomial_predictions = []
            wc_monomials = []
            for prediction_function, data in zip(prediction_functions, prediction_data):
                polynomial_prediction, wc_monomial = prediction_function(
                    wc_array, scale, data
                )
                polynomial_predictions.append(polynomial_prediction)
                wc_monomials.append(wc_monomial)
            polynomial_predictions = jnp.concatenate(polynomial_predictions, axis=-1)
            return polynomial_predictions, wc_monomials

        return prediction

    def _get_prediction_function_no_theory_uncertainty(self):

        prediction_functions = [
            ObservableSector.get(name).prediction
            for name in self.observable_sectors_no_theory_uncertainty
        ]

        @jit
        def prediction(
            wc_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data: List[List[jnp.array]]
        ) -> jnp.array:
            return jnp.concatenate([
                prediction_function(wc_array, scale, data)[0]
                for prediction_function, data in zip(prediction_functions, prediction_data)
            ], axis=-1)

        return prediction

    def _get_constraints_no_theory_uncertainty(self):
        constraints = Measurement.get_constraints(self.observables_no_theory_uncertainty)
        constraint_dict = {}

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

        # multivariate normal distribution
        if 'MultivariateNormalDistribution' in constraints:
            len_mvnd = len(constraints['MultivariateNormalDistribution']['observables'])
            constraint_dict['MultivariateNormalDistribution'] = [
                [
                    jnp.asarray(constraints['MultivariateNormalDistribution']['observable_indices'][i])
                    for i in range(len_mvnd)
                ],
                [
                    jnp.asarray(constraints['MultivariateNormalDistribution']['central_value'][i])
                    for i in range(len_mvnd)
                ],
                [
                    jnp.asarray(constraints['MultivariateNormalDistribution']['standard_deviation'][i])
                    for i in range(len_mvnd)
                ],
                [
                    jnp.asarray(constraints['MultivariateNormalDistribution']['inverse_correlation'][i])
                    for i in range(len_mvnd)
                ],
                [
                    jnp.asarray(constraints['MultivariateNormalDistribution']['logpdf_normalization_per_observable'][i])
                    for i in range(len_mvnd)
                ],
            ]

        return constraint_dict

    def _get_logpdf_function_no_theory_uncertainty(self):

        prediction_function_no_theory_uncertainty = self.prediction_function_no_theory_uncertainty

        @jit
        def logpdf_no_theory_uncertainty(
            wc_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data_no_theory_uncertainty: jnp.array,
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
        ) -> jnp.array:
            predictions = prediction_function_no_theory_uncertainty(wc_array, scale, prediction_data_no_theory_uncertainty)
            logpdf = jnp.zeros_like(predictions)
            for distribution_type in constraints_no_theory_uncertainty.keys():
                logpdf += logpdf_functions[distribution_type](
                    predictions,
                    *constraints_no_theory_uncertainty[distribution_type]
                )
            return logpdf

        return logpdf_no_theory_uncertainty

    def _get_global_logpdf_function_no_theory_uncertainty(self):
        logpdf_function_no_theory_uncertainty = self.logpdf_function_no_theory_uncertainty

        @jit
        def global_logpdf_no_theory_uncertainty(
            wc_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data_no_theory_uncertainty: List[List[jnp.array]],
            constraints_no_theory_uncertainty: Dict[str,Union[List[jnp.array],List[List[jnp.array]]]],
        ) -> jnp.array:
            return jnp.sum(
                logpdf_function_no_theory_uncertainty(
                    wc_array, scale, prediction_data_no_theory_uncertainty,
                    constraints_no_theory_uncertainty
                ),
                axis=0
            )

        return global_logpdf_no_theory_uncertainty
