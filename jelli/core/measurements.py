import json
import os
import numpy as np
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Dict, List, Set, Optional
from ..utils.data_io import pad_arrays
from ..utils.distributions import convert_GeneralGammaDistributionPositive, LOG_ZERO
from ..utils.types import ObservablesType
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
        - observables : list containing str or tuple
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
        - observables : list containing str or tuple
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
            - observables : list containing str or tuple
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

            # Change 'type' to 'distribution_type' (to avoid conflict with python built-in type)
            # TODO: Change 'type' to 'distribution_type' in the json files
            constraint['distribution_type'] = constraint['type']
            del constraint['type']

            # Convert list of observable names to numpy array containing strings
            constraint['observables'] = np.array([
                obs if isinstance(obs, str) else obs[0] if len(obs)==1 else str(tuple(obs))
                for obs in constraint['observables']
            ], dtype=object)

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
            std = np.array(parameters['standard_deviation'])
            corr = np.array(parameters['correlation'])
            parameters['covariance'] = np.outer(std, std) * corr
            del parameters['standard_deviation']
            del parameters['correlation']

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
    def get_measurements(cls, observables: ObservablesType) -> Dict[str, 'Measurement']:
        '''
        Return measurements that constrain the specified observables.

        Parameters
        ----------
        observables : list or array of str
            Observables to constrain

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
        measurement_names = set(chain.from_iterable(cls._observable_to_measurements[o] for o in observables))
        return {name: cls._measurements[name] for name in measurement_names}

    @classmethod
    def get_constraints(
        cls,
        observables: ObservablesType,
        distribution_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        '''
        Return constraints on the specified observables.

        Parameters
        ----------
        observables : list or array of str or tuple
            Observables to constrain
        distribution_types : list of str, optional
            Types of distributions to include. If None, include all distributions.

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
        measurements = cls.get_measurements(observables)
        observables_set = set(observables)
        constraints = defaultdict(lambda: defaultdict(list))
        for measurement in measurements.values():
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
                        constraint_covariance = np.array(constraint['parameters']['covariance'])[mask][:, mask]
                        observable_indices = np.array([observables.index(obs) for obs in constraint_observables])

                        if np.sum(mask) == 1: # Univariate normal distribution
                            constraints['NormalDistribution']['observables'].extend(constraint_observables)
                            constraints['NormalDistribution']['observable_indices'].extend(observable_indices)
                            constraints['NormalDistribution']['central_value'].extend(constraint_central_value)
                            constraints['NormalDistribution']['standard_deviation'].append(
                                np.sqrt(constraint_covariance[0, 0])
                            )
                        else: # Multivariate normal distribution
                            constraints[distribution_type]['observables'].append(constraint_observables)
                            constraints[distribution_type]['observable_indices'].append(observable_indices)
                            constraints[distribution_type]['central_value'].append(constraint_central_value)
                            constraints[distribution_type]['covariance'].append(constraint_covariance)
                            constraints[distribution_type]['inverse_covariance'].append(
                                np.linalg.inv(constraint_covariance)
                            )
                    else:
                        observable_indices = [observables.index(obs) for obs in constraint['observables']]
                        constraints[distribution_type]['observables'].extend(constraint['observables'])
                        constraints[distribution_type]['observable_indices'].extend(observable_indices)
                        for key in constraint['parameters']:
                            constraints[distribution_type][key].append(constraint['parameters'][key])
        for distribution_type in constraints:

            # Convert list of observable names to numpy array containing strings
            constraints[distribution_type]['observables'] = np.array(
                constraints[distribution_type]['observables'],
                dtype=object
            )

            # Pad arrays to the same length for numerical distributions
            if distribution_type == 'NumericalDistribution':
                constraints[distribution_type]['x'] = pad_arrays(constraints[distribution_type]['x'])
                constraints[distribution_type]['y'] = pad_arrays(constraints[distribution_type]['y'])
                constraints[distribution_type]['log_y'] = pad_arrays(constraints[distribution_type]['log_y'])

            # Convert lists to numpy arrays
            dtype = object if distribution_type == 'MultivariateNormalDistribution' else None
            for key in constraints[distribution_type]:
                constraints[distribution_type][key] = np.asarray(
                    constraints[distribution_type][key],
                    dtype=dtype,
                )
        return constraints

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
