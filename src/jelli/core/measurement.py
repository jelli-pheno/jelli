import json
import os
import numpy as np
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Dict, List, Set, Optional, Union
from ..utils.data_io import pad_arrays
from ..utils.distributions import convert_GeneralGammaDistributionPositive, LOG_ZERO, combine_normal_distributions, combine_distributions_numerically
from ..utils.data_io import get_json_schema

class Measurement:
    '''
    Class to store measurements and constraints on observables.

    Parameters
    ----------
    name : str
        Name of the measurement.
    constraints : list of dict
        List of constraints on observables. Each constraint is a dictionary with the following keys:
        - type : str
            Type of the distribution. Can be 'NormalDistribution', 'HalfNormalDistribution', 'GammaDistributionPositive', 'NumericalDistribution', 'MultivariateNormalDistribution'.
        - observables : list containing str
            List of observables that the constraint applies to. The observables are either strings or tuples (in case of additional arguments like a $q^2$ value or $q^2$ range).
        - parameters : dict
            Parameters of the distribution. The keys depend on the type of the distribution as follows:
            - NormalDistribution : 'central_value' (float), 'standard_deviation' (float)
            - HalfNormalDistribution : 'central_value' (float), 'standard_deviation' (float)
            - GammaDistributionPositive : 'a' (float), 'loc' (float), 'scale' (float)
            - NumericalDistribution : 'x' (list of float), 'y' (list of float)
            - MultivariateNormalDistribution : 'central_value' (list of float), 'covariance' (list of list of float)

    Attributes
    ----------
    name : str
        Name of the measurement.
    constraints : list of dict
        List of constraints on observables. Each constraint is a dictionary with the following keys:
        - observables : list containing str
            List of observables that the constraint applies to. The observables are either strings or tuples (in case of additional arguments like a $q^2$ value or $q^2$ range).
        - distribution_type : str
            Type of the distribution. Can be 'NormalDistribution', 'HalfNormalDistribution', 'GammaDistributionPositive', 'NumericalDistribution', 'MultivariateNormalDistribution'.
        - parameters : dict
            Parameters of the distribution. The keys depend on the type of the distribution as follows:
            - NormalDistribution : 'central_value' (float), 'standard_deviation' (float)
            - HalfNormalDistribution : 'central_value' (float), 'standard_deviation' (float)
            - GammaDistributionPositive : 'a' (float), 'loc' (float), 'scale' (float)
            - NumericalDistribution : 'x' (list of float), 'y' (list of float)
            - MultivariateNormalDistribution : 'central_value' (list of float), 'covariance' (list of list of float)
    constrained_observables : set
        Set of observables that the measurement constrains

    Methods
    -------
    get_all_measurements()
        Return all measurements.
    get_all_observables()
        Return all observables.
    get_measurements(observables)
        Return measurements that constrain the specified observables.
    get_constraints(observables)
        Return constraints on the specified observables.
    get_combined_constraints(observables)
        Return combined constraints on the specified observables.
    load(path)
        Load measurements from a json file or a directory containing json files
    unload(measurement_names)
        Unload measurements.
    clear()
        Clear all measurements.

    Examples
    --------
    Load measurements from a json file:

    >>> Measurement.load('measurements.json')

    Get all measurements:

    >>> Measurement.get_all_measurements()
    {'measurement1': <Measurement object>, 'measurement2': <Measurement object>, ...}

    Get all observables:

    >>> Measurement.get_all_observables()
    {'observable1', 'observable2', ...}

    Get measurements that contain the specified observables:

    >>> Measurement.get_measurements({'observable1', 'observable2'})
    {'measurement1': <Measurement object>, 'measurement2': <Measurement object>, ...}

    Get constraints on the specified observables:

    >>> Measurement.get_constraints({'observable1', 'observable2'})
    {'NormalDistribution': {'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'standard_deviation': [1.0, 1.0]}, ...}

    Get combined constraints on the specified observables:

    >>> Measurement.get_combined_constraints({'observable1', 'observable2'})
    {'NormalDistribution': {'measurement_name': ['measurement1'], 'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'standard_deviation': [1.0, 1.0]}, ...}

    Unload measurements:

    >>> Measurement.unload(['measurement1', 'measurement2'])

    Clear all measurements:

    >>> Measurement.clear()
    '''

    _measurements: Dict[str, 'Measurement'] = {}  # Class attribute to store all measurements
    _observable_to_measurements: DefaultDict[str, Set[str]] = defaultdict(set)  # Class attribute to map observables to measurements
    _pdfxf_versions = ['1.0'] # List of supported versions of the pdfxf JSON schema

    def __init__(self, name: str, constraints: List[dict]):
        '''
        Initialize a Measurement object.

        Parameters
        ----------
        name : str
            Name of the measurement.
        constraints : list of dict
            List of constraints on observables. Each constraint is a dictionary with the following keys:
            - type : str
                Type of the distribution. Can be 'NormalDistribution', 'HalfNormalDistribution', 'GammaDistributionPositive', 'NumericalDistribution', 'MultivariateNormalDistribution'.
            - observables : list containing str
                List of observables that the constraint applies to. The observables are either strings or tuples (in case of additional arguments like a $q^2$ value or $q^2$ range).
            - parameters : dict
                Parameters of the distribution. The keys depend on the type of the distribution as follows:
                - NormalDistribution : 'central_value' (float), 'standard_deviation' (float)
                - HalfNormalDistribution : 'central_value' (float), 'standard_deviation' (float)
                - GammaDistributionPositive : 'a' (float), 'loc' (float), 'scale' (float)
                - NumericalDistribution : 'x' (list of float), 'y' (list of float)
                - MultivariateNormalDistribution : 'central_value' (list of float), 'covariance' (list of list of float)

        Returns
        -------
        None

        Examples
        --------
        Initialize a Measurement object:

        >>> Measurement('measurement1', [{'type': 'NormalDistribution', 'observables': ['observable1'], 'parameters': {'central_value': 0.0, 'standard_deviation': 1.0}}])
        '''
        self.name: str = name
        self.constraints: list[dict] = []
        for constraint in constraints:

            # Convert list of observable names to numpy array containing strings
            constraint['observables'] = np.array(constraint['observables'], dtype=object)

            # Add measurement name to `_observable_to_measurements` class attribute
            for observable in constraint['observables']:
                self._observable_to_measurements[observable].add(name)

            # Add constraint to `constraints` attribute of the Measurement object
            self.constraints.append(self._define_constraint(**constraint))

        # Add set of observables that the measurement constrains to `constrained_observables` attribute of the Measurement object
        self.constrained_observables = set(chain.from_iterable(
            constraint['observables'] for constraint in self.constraints
        ))

        # Add measurement to `_measurements` class attribute
        self._measurements[name] = self

    @staticmethod
    def _define_constraint(observables: np.ndarray, distribution_type: str, **parameters: dict) -> dict:

        # Convert GeneralGammaDistributionPositive to NumericalDistribution
        if distribution_type == 'GeneralGammaDistributionPositive':
            distribution_type, parameters = convert_GeneralGammaDistributionPositive(**parameters)

        # Convert lists to numpy arrays for numerical distributions,
        # normalize PDF to 1, and add log PDF
        elif distribution_type == 'NumericalDistribution':
            x = np.array(parameters['x'])
            y = np.array(parameters['y'])
            y = np.maximum(0, y)  # make sure PDF is positive
            y = y /  np.trapz(y, x=x)  # normalize PDF to 1
            # ignore warning from log(0)=-np.inf
            with np.errstate(divide='ignore', invalid='ignore'):
                log_y = np.log(y)
            # replace -np.inf with a large negative number
            log_y[np.isneginf(log_y)] = LOG_ZERO
            parameters['x'] = x
            parameters['y'] = y
            parameters['log_y'] = log_y

        # Convert lists to numpy arrays for multivariate normal distribution
        elif distribution_type == 'MultivariateNormalDistribution':
            parameters['standard_deviation'] = np.array(parameters['standard_deviation'])
            parameters['correlation'] = np.array(parameters['correlation'])

        return {'observables': observables, 'distribution_type': distribution_type, 'parameters': parameters}

    def __repr__(self):
        return f'<Measurement {self.name} constraining {self.constrained_observables}>'

    def __str__(self):
        return f'Measurement {self.name} constraining {self.constrained_observables}'

    @classmethod
    def get_all_measurements(cls):
        '''
        Return all measurements.

        Returns
        -------
        dict
            Dictionary containing all measurements.

        Examples
        --------
        Get all measurements:

        >>> Measurement.get_all_measurements()
        {'measurement1': <Measurement object>, 'measurement2': <Measurement object>, ...}
        '''
        return cls._measurements

    @classmethod
    def get_all_observables(cls):
        '''
        Return all observables.

        Returns
        -------
        set
            Set containing all observables.

        Examples
        --------
        Get all observables:

        >>> Measurement.get_all_observables()
        {'observable1', 'observable2', ...}
        '''
        return set(cls._observable_to_measurements.keys())

    @classmethod
    def get_measurements(
        cls,
        observables: Union[List[str], np.ndarray],
        include_measurements: Optional[List[str]] = None,
        exclude_measurements: Optional[List[str]] = None,
    ) -> Dict[str, 'Measurement']:
        '''
        Return measurements that constrain the specified observables.

        Parameters
        ----------
        observables : list or array of str
            Observables to constrain
        include_measurements : list of str, optional
            A list of measurements to include. If None, include all measurements.
        exclude_measurements : list of str, optional
            A list of measurements to exclude. If None, exclude no measurements.

        Returns
        -------
        dict
            Dictionary containing measurements that constrain the specified observables.

        Examples
        --------
        Get measurements that constrain the specified observables:

        >>> Measurement.get_measurements(['observable1', 'observable2'])
        {'measurement1': <Measurement object>, 'measurement2': <Measurement object>, ...}
        '''
        if include_measurements is not None and exclude_measurements is not None:
            raise ValueError("Please provide either `include_measurements` or `exclude_measurements`, not both.")
        measurement_names = set(chain.from_iterable(
            cls._observable_to_measurements.get(observable, set())
            for observable in observables
        ))
        all_measurements = set(cls.get_all_measurements())
        if include_measurements is not None:
            if set(include_measurements) - all_measurements:
                raise ValueError(f"Measurements {set(include_measurements) - all_measurements} provided in `include_measurements` not found in loaded measurements.")
            measurement_names = set(include_measurements) & measurement_names
        elif exclude_measurements is not None:
            if set(exclude_measurements) - all_measurements:
                raise ValueError(f"Measurements {set(exclude_measurements) - all_measurements} provided in `exclude_measurements` not found in loaded measurements.")
            measurement_names = measurement_names - set(exclude_measurements)
        return {name: cls._measurements[name] for name in measurement_names}

    @classmethod
    def get_constraints(
        cls,
        observables: Union[List[str], np.ndarray],
        observables_for_indices: Union[List[str], np.ndarray] = None,
        distribution_types: Optional[List[str]] = None,
        include_measurements: Optional[List[str]] = None,
        exclude_measurements: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        '''
        Return constraints on the specified observables.

        Parameters
        ----------
        observables : list or array of str
            Observables to constrain
        observables_for_indices : list or array of str, optional
            Observables to create indices for. If None, use the same observables as `observables`.
        distribution_types : list of str, optional
            Types of distributions to include. If None, include all distributions.
        include_measurements : list of str, optional
            A list of measurements to include. If None, include all measurements.
        exclude_measurements : list of str, optional
            A list of measurements to exclude. If None, exclude no measurements.

        Returns
        -------
        dict
            Dictionary containing constraints on the specified observables.

        Examples
        --------
        Get constraints on the specified observables:

        >>> Measurement.get_constraints(['observable1', 'observable2'])
        {'NormalDistribution': {'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'standard_deviation': [1.0, 1.0]}, ...}

        Get constraints on the specified observables with specific distribution types:

        >>> Measurement.get_constraints(['observable1', 'observable2'], ['NormalDistribution', 'MultivariateNormalDistribution'])
        {'NormalDistribution': {'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'standard_deviation': [1.0, 1.0]}, 'MultivariateNormalDistribution': {'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'covariance': [[1.0, 0.0], [0.0, 1.0]], 'inverse_covariance': [[1.0, 0.0], [0.0, 1.0]]}}
        '''
        if not isinstance(observables, (list, tuple, np.ndarray)):
            raise ValueError('observables must be a list, tuple, or array')
        if isinstance(observables, np.ndarray):
            observables = observables.tolist()
        if observables_for_indices is None:
            observables_for_indices = observables
        else:
            if not isinstance(observables_for_indices, (list, tuple, np.ndarray)):
                raise ValueError('observables_for_indices must be a list, tuple, or array')
            if isinstance(observables_for_indices, np.ndarray):
                observables_for_indices = observables_for_indices.tolist()
        measurements = cls.get_measurements(observables, include_measurements, exclude_measurements)
        observables_set = set(observables)
        constraints = defaultdict(lambda: defaultdict(list))
        for measurement_name, measurement in measurements.items():
            for constraint in measurement.constraints:
                selected_observables = set(constraint['observables']) & observables_set
                if selected_observables:
                    distribution_type = constraint['distribution_type']
                    if distribution_types is not None and distribution_type not in distribution_types:
                        continue
                    if distribution_type == 'MultivariateNormalDistribution':

                        # Boolean mask for order-preserving selection
                        selected_observables_array = np.array(list(selected_observables), dtype=object)
                        mask = np.isin(constraint['observables'], selected_observables_array)

                        # Skip if no matches
                        if not np.any(mask):
                            continue

                        # Select entries using the boolean mask
                        constraint_observables = constraint['observables'][mask]
                        constraint_central_value = np.array(constraint['parameters']['central_value'])[mask]
                        constraint_standard_deviation = np.array(constraint['parameters']['standard_deviation'])[mask]
                        constraint_correlation = np.array(constraint['parameters']['correlation'])[mask][:, mask]
                        observable_indices = np.array([observables_for_indices.index(obs) for obs in constraint_observables])

                        if np.sum(mask) == 1: # Univariate normal distribution
                            constraints['NormalDistribution']['measurement_name'].append(measurement_name)
                            constraints['NormalDistribution']['observables'].extend(constraint_observables)
                            constraints['NormalDistribution']['observable_indices'].extend(observable_indices)
                            constraints['NormalDistribution']['central_value'].extend(constraint_central_value)
                            constraints['NormalDistribution']['standard_deviation'].extend(constraint_standard_deviation)
                        else: # Multivariate normal distribution
                            constraints[distribution_type]['measurement_name'].append(measurement_name)
                            constraints[distribution_type]['observables'].append(
                                np.asarray(constraint_observables, dtype=object)
                            )
                            constraints[distribution_type]['observable_indices'].append(
                                np.asarray(observable_indices, dtype=int)
                            )
                            constraints[distribution_type]['central_value'].append(
                                np.asarray(constraint_central_value)
                            )
                            constraints[distribution_type]['standard_deviation'].append(
                                np.asarray(constraint_standard_deviation)
                            )
                            constraints[distribution_type]['inverse_correlation'].append(
                                np.linalg.inv(constraint_correlation)
                            )
                            n = len(constraint_observables)
                            logdet_corr = np.linalg.slogdet(constraint_correlation)[1]
                            logprod_std2 = 2 * np.sum(np.log(constraint_standard_deviation))
                            constraints[distribution_type]['logpdf_normalization_per_observable'].append(
                                -0.5 * ( (logdet_corr + logprod_std2) / n + np.log(2 * np.pi) )
                            )
                    else:
                        constraints[distribution_type]['measurement_name'].append(measurement_name)
                        observable_indices = [observables_for_indices.index(obs) for obs in constraint['observables']]
                        constraints[distribution_type]['observables'].extend(constraint['observables'])
                        constraints[distribution_type]['observable_indices'].extend(observable_indices)
                        for key in constraint['parameters']:
                            constraints[distribution_type][key].append(constraint['parameters'][key])
        for distribution_type in constraints:

            # Pad arrays to the same length for numerical distributions
            if distribution_type == 'NumericalDistribution':
                constraints[distribution_type]['x'] = pad_arrays(constraints[distribution_type]['x'])
                constraints[distribution_type]['y'] = pad_arrays(constraints[distribution_type]['y'])
                constraints[distribution_type]['log_y'] = pad_arrays(constraints[distribution_type]['log_y'])

            # Convert lists to numpy arrays
            if distribution_type == 'MultivariateNormalDistribution':
                for key in constraints[distribution_type]:
                    nparray = np.empty(len(constraints[distribution_type][key]), dtype=object)
                    nparray[:] = constraints[distribution_type][key]
                    constraints[distribution_type][key] = nparray
            else:
                for key in constraints[distribution_type]:
                    if key == 'observable_indices':
                        dtype = int
                    elif key == 'observables':
                        dtype = object
                    else:
                        dtype = None
                    constraints[distribution_type][key] = np.asarray(
                        constraints[distribution_type][key],
                        dtype=dtype
                    )
        return constraints

    def combine_constraints(
            constraints_list: List[Dict[str, Dict[str, np.ndarray]]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        '''
        Combine the constraints provided in the list of constraints, where each element of the list is a dictionary of constraints on a single observable.

        Normal distributions are combined analytically, while other distributions are combined numerically.

        Parameters
        ----------
        constraints_list : list of dict
            List of constraints to combine, one constraints dictionary per observable.

        Returns
        -------
        dict
            Dictionary containing combined constraints.

        Examples
        --------
        Combine two constraints on two observables:
        >>> Measurement.combine_constraints([
        ...     {'NormalDistribution': {'measurement_name': ['measurement1', 'measurement2'], 'observables': ['observable1', 'observable1'], 'observable_indices': np.array([0, 0]), 'central_value': np.array([1.0, 1.2]), 'standard_deviation': np.array([0.2, 0.3])}},
        ...     {'NormalDistribution': {'measurement_name': ['measurement3', 'measurement4'], 'observables': ['observable2', 'observable2'], 'observable_indices': np.array([1, 1]), 'central_value': np.array([2.0, 2.5]), 'standard_deviation': np.array([0.5, 0.7])}},
        ... ])
        {'NormalDistribution':
            {
                'measurement_name': array(['measurement1, measurement2', 'measurement3, measurement4']),
                'observables': array(['observable1', 'observable2']),
                'observable_indices': array([0, 1]),
                'central_value': array([1.06153846, 2.16891892]),
                'standard_deviation': array([0.16641006, 0.40686674])
            },
        }
        '''

        combined_constraints = defaultdict(lambda: defaultdict(list))
        for constraints in constraints_list:
            # handle normal distributions
            if 'NormalDistribution' in constraints:
                constraints['NormalDistribution'] = combine_normal_distributions(**constraints['NormalDistribution'])

            if len(constraints) > 1 or len(next(iter(constraints.values()))['measurement_name']) > 1:
                numerical_distribution = combine_distributions_numerically(constraints)
                for key, value in numerical_distribution.items():
                    combined_constraints['NumericalDistribution'][key].append(value)
            else:
                dist_type, dist_info = next(iter(constraints.items()))
                for key, value in dist_info.items():
                    if dist_type == 'NumericalDistribution' and key in ['x', 'y', 'log_y']:
                        value = np.squeeze(value)
                    combined_constraints[dist_type][key].append(value)

        for dist_type, dist_info in combined_constraints.items():
            for key, value_list in dist_info.items():
                if dist_type == 'NumericalDistribution' and key in ['x', 'y', 'log_y']:
                    combined_constraints[dist_type][key] = pad_arrays(value_list)
                else:
                    combined_constraints[dist_type][key] = np.concatenate(value_list, axis=0)
        return combined_constraints

    @classmethod
    def get_combined_constraints(
        cls,
        observables: Union[List[str], np.ndarray],
        ) -> Dict[str, Dict[str, np.ndarray]]:
        '''
        Return combined constraints on the specified observables.

        Parameters
        ----------
        observables : list or array of str
            Observables to combine constraints for.

        Returns
        -------
        dict
            Dictionary containing combined constraints on the specified observables.

        Examples
        --------
        Get combined constraints on the specified observables:

        >>> Measurement.get_combined_constraints(['observable1', 'observable2'])
        {'NormalDistribution': {'measurement_name': ['measurement1'], 'observables': ['observable1', 'observable2'], 'observable_indices': [0, 1], 'central_value': [0.0, 0.0], 'standard_deviation': [1.0, 1.0]}, ...}
        '''

        return cls.combine_constraints(
            [cls.get_constraints([observable], observables) for observable in observables]
        )

    @classmethod
    def _load_file(cls, path: str) -> None:
        with open(path, 'r') as f:
            json_data = json.load(f)
        schema_name, schema_version = get_json_schema(json_data)
        if schema_name == 'pdfxf' and schema_version in cls._pdfxf_versions:
            del json_data['$schema']
            for name, constraints in json_data.items():
                cls(name, constraints)

    @classmethod
    def load(cls, path: str) -> None:
        '''
        Load measurements from a json file or a directory containing json files.

        Parameters
        ----------
        path : str
            Path to a json file or a directory containing json files.

        Returns
        -------
        None

        Examples
        --------
        Load measurements from a json file:

        >>> Measurement.load('./measurements.json')

        Load measurements from a directory containing json files:

        >>> Measurement.load('./measurements/')
        '''
        # load all json files in the directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.json'):
                    cls._load_file(os.path.join(path, file))
        # load single json file
        else:
            cls._load_file(path)

    @classmethod
    def unload(cls, measurement_names: List[str]) -> None:
        '''
        Unload measurements.

        Parameters
        ----------
        measurement_names : list of str
            Names of the measurements to unload.

        Returns
        -------
        None

        Examples
        --------
        Unload measurements:

        >>> Measurement.unload(['measurement1', 'measurement2'])
        '''
        for name in measurement_names:
            measurement = cls._measurements.pop(name, None)
            if measurement is not None:
                for constraint in measurement.constraints:
                    for observable in constraint['observables']:
                        cls._observable_to_measurements[observable].remove(name)
                        if not cls._observable_to_measurements[observable]:
                            del cls._observable_to_measurements[observable]

    @classmethod
    def clear(cls) -> None:
        '''
        Clear all measurements.

        Returns
        -------
        None

        Examples
        --------
        Clear all measurements:

        >>> Measurement.clear()
        '''
        cls._measurements.clear()
        cls._observable_to_measurements.clear()
