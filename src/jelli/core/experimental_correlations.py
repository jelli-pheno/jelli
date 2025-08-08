from typing import Dict, Optional, Iterable
import numpy as np
from jax import numpy as jnp
from jelli.utils.data_io import hash_names
from jelli.utils.distributions import logpdf_functions, get_distribution_samples
from jelli.core.measurement import Measurement
from jelli.core.observable_sector import ObservableSector
from itertools import chain

class ExperimentalCorrelations:

    _instances: Dict[str, Dict[str, 'ExperimentalCorrelations']] = {
        'correlations': {},
        'central': {},
        'uncertainties': {}
    }  # Dictionary to hold instances of ExperimentalCorrelations
    _covariance_scaled: Dict[str, jnp.ndarray] = {}
    _observable_names: Iterable[Iterable[str]] = []

    def __init__(
        self,
        hash_val: str,
        data_type: str,
        data: np.ndarray,
        row_names: Iterable[str],
        col_names: Iterable[str],
    ) -> None:
        self.hash_val = hash_val
        self.data_type = data_type
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self._instances[data_type][hash_val] = self

    @classmethod
    def load(cls) -> None:
        observable_names = []
        for observable_sector in ObservableSector.get_all():
            if observable_sector.observable_uncertainties is not None:
                observable_names.append(observable_sector.observable_names)
        cls._observable_names = observable_names
        cls._instances = {
            'correlations': {},
            'central': {},
            'uncertainties': {}
        }
        cls._covariance_scaled = {}

    @classmethod
    def compute(
        cls,
        include_measurements: Iterable[str],
        n_samples: int = int(1e6),
        seed: int = None
    ) -> None:

        observables = list(chain.from_iterable(cls._observable_names))

        # get univariate constraints and combine them for each observable
        constraints_univariate = Measurement.get_constraints(
            observables,
            distribution_types=[k for k in logpdf_functions.keys() if k not in ['MultivariateNormalDistribution']],
            include_measurements=include_measurements,
            )
        constraints_list = []
        for observable in observables:
            constraints_observable = {}
            for dist_type, dist_info in constraints_univariate.items():
                mask = dist_info['observables'] == observable
                if np.any(mask):
                    constraints_observable[dist_type] = {
                        k: v[mask] for k, v in dist_info.items()
                    }
            if constraints_observable:
                constraints_list.append(constraints_observable)
        constraints_univariate = Measurement.combine_constraints(constraints_list)

        # construct covariance matrix and mean vector from univariate constraints
        cov = np.diag([np.inf] * len(observables))
        mean = np.zeros(len(observables))
        for dist_type, dist_info in constraints_univariate.items():
            if dist_type == 'NormalDistribution' or dist_type == 'HalfNormalDistribution':  # replace HalfNormalDistribution with zero-mean NormalDistribution
                central_value = dist_info['central_value']
                standard_deviation = dist_info['standard_deviation']
                observable_indices = dist_info['observable_indices']
            else:  # numerically obtain the Gaussian approximation
                samples = get_distribution_samples(dist_type, dist_info, n_samples, seed)
                central_value = np.mean(samples, axis=1)
                standard_deviation = np.std(samples, axis=1)
                observable_indices = dist_info['observable_indices']
            cov[observable_indices, observable_indices] = standard_deviation**2
            mean[observable_indices] = central_value

        # get multivariate constraints
        constraints_multivariate = Measurement.get_constraints(
            observables,
            distribution_types=['MultivariateNormalDistribution'],
            include_measurements=include_measurements,
        )
        if constraints_multivariate:
            # combine all covariance matrices and mean vectors using the weighted average
            weights = [np.diag(1/np.diag(cov))]
            means = [mean]
            constraints_multivariate = constraints_multivariate['MultivariateNormalDistribution']
            for i in range(len(constraints_multivariate['central_value'])):
                weight_i = np.zeros((len(observables), len(observables)))
                mean_i = np.zeros(len(observables))
                observable_indices_i = constraints_multivariate['observable_indices'][i]
                central_value_i = constraints_multivariate['central_value'][i]
                standard_deviation_i = constraints_multivariate['standard_deviation'][i]
                inverse_correlation_i = constraints_multivariate['inverse_correlation'][i]
                weight_i[np.ix_(observable_indices_i, observable_indices_i)] = inverse_correlation_i / np.outer(standard_deviation_i, standard_deviation_i)
                mean_i[observable_indices_i] = central_value_i
                weights.append(weight_i)
                means.append(mean_i)
            inv_cov = np.sum(weights, axis=0)
            # regularize inversion asuming inv_cov = D R D where R has unit diagonal, then invert R instead of inv_cov
            d = np.sqrt(np.diag(inv_cov))
            nonzero = d != 0  # unconstrained observables have zeros
            inv_cov = inv_cov[np.ix_(nonzero, nonzero)]
            d = d[nonzero]
            d2 = np.outer(d, d)
            R = inv_cov / d2
            inv_R = np.linalg.inv(R)
            cov = np.diag([np.nan] * len(nonzero))
            cov[np.ix_(nonzero, nonzero)] = inv_R / d2
            mean = cov @ np.sum([w @ m for w, m in zip(weights, means)], axis=0)
        else:
            cov[cov == np.inf] = np.nan
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)

        for i, row_names in enumerate(cls._observable_names):
            row_measurements = Measurement.get_measurements(row_names, include_measurements=include_measurements)
            row_idx = [observables.index(o) for o in row_names]
            hash_val = hash_names(row_measurements, row_names)
            cls(
                hash_val=hash_val,
                data_type='central',
                data=jnp.array(mean[row_idx], dtype=jnp.float64),
                row_names=row_names,
                col_names=[],
            )
            cls(
                hash_val=hash_val,
                data_type='uncertainties',
                data=jnp.array(std[row_idx], dtype=jnp.float64),
                row_names=row_names,
                col_names=[],
            )
            for j in range(i, len(cls._observable_names)):
                col_names = cls._observable_names[j]
                col_measurements = Measurement.get_measurements(col_names, include_measurements=include_measurements)
                col_idx = [observables.index(o) for o in col_names]
                hash_val = hash_names(row_measurements, col_measurements, row_names, col_names)
                cls(
                    hash_val=hash_val,
                    data_type='correlations',
                    data=jnp.array(corr[np.ix_(row_idx, col_idx)], dtype=jnp.float64),
                    row_names=row_names,
                    col_names=col_names,
                )

    @classmethod
    def get_data(
        cls,
        data_type: str,
        include_measurements: Iterable[str],
        row_names: Iterable[str],
        col_names: Optional[Iterable[str]] = []
    ):
        row_measurements = Measurement.get_measurements(row_names, include_measurements=include_measurements)
        col_measurements = Measurement.get_measurements(col_names, include_measurements=include_measurements)
        hash_val = hash_names(row_measurements, col_measurements, row_names, col_names)
        hash_val_transposed = hash_names(col_measurements, row_measurements, col_names, row_names)
        if hash_val not in cls._instances[data_type] and hash_val_transposed not in cls._instances[data_type]:
            cls.compute(include_measurements)
        if hash_val in cls._instances[data_type]:
            return cls._instances[data_type][hash_val].data
        elif hash_val_transposed in cls._instances[data_type]:
            return cls._instances[data_type][hash_val_transposed].data.T
        else:
            return None

    @classmethod
    def get_cov_scaled(
        cls,
        include_measurements: Iterable[str],
        row_names: Iterable[str],
        col_names: Iterable[str],
        std_exp_scaled_row: np.ndarray,
        std_exp_scaled_col: np.ndarray,
    ):
        row_measurements = Measurement.get_measurements(row_names, include_measurements=include_measurements)
        col_measurements = Measurement.get_measurements(col_names, include_measurements=include_measurements)
        hash_val = hash_names(row_measurements, col_measurements, row_names, col_names)
        if hash_val in cls._covariance_scaled:
            cov_scaled = cls._covariance_scaled[hash_val]
        else:
            corr = cls.get_data('correlations', include_measurements, row_names, col_names)
            if corr is None:
                raise ValueError(f"Correlation data for {row_names} and {col_names} not found.")
            cov_scaled = corr * np.outer(std_exp_scaled_row, std_exp_scaled_col)
            cov_scaled = jnp.array(cov_scaled, dtype=jnp.float64)
            cls._covariance_scaled[hash_val] = cov_scaled
        return cov_scaled
