from typing import Dict, Iterable
import h5py
import os
import numpy as np
from jax import numpy as jnp
from jelli.utils.data_io import get_json_schema, hash_names
from jelli.core.measurement import Measurement


class TheoryCorrelations:
    '''
    A class to represent theory correlations.

    Parameters
    ----------
    hash_val : str
        The hash value representing the combination of row and column observable names.
    data : np.ndarray
        The data array containing the correlation data.
    row_names : Dict[str, int]
        A dictionary mapping row observable names to their indices.
    col_names : Dict[str, int]
        A dictionary mapping column observable names to their indices.

    Attributes
    ----------
    hash_val : str
        The hash value representing the combination of row and column observable names.
    data : np.ndarray
        The data array containing the correlation data.
    row_names : Dict[str, int]
        A dictionary mapping row observable names to their indices.
    col_names : Dict[str, int]
        A dictionary mapping column observable names to their indices.
    _correlations : Dict[str, 'TheoryCorrelations']
        A class attribute to cache all theory correlations.
    _covariance_scaled : Dict[str, jnp.ndarray]
        A class attribute to cache scaled covariance matrices.
    _popxf_h5_versions : Set[str]
        A set of supported versions of the popxf-h5 JSON schema.

    Methods
    -------
    load(path: str) -> None
        Load theory correlations from HDF5 files in the specified path.
    _load_file(path: str) -> None
        Load theory correlations from a single HDF5 file.
    from_hdf5_group(hash_val: str, hdf5_group: h5py.Group) -> None
        Create a TheoryCorrelations instance from an HDF5 group.
    get_data(row_names: Iterable[str], col_names: Iterable[str]) -> np.ndarray or None
        Get the correlation data for the specified row and column observable names.
    get_cov_scaled(
        include_measurements: Iterable[str],
        row_names: Iterable[str],
        col_names: Iterable[str],
        std_th_scaled_row: np.ndarray,
        std_th_scaled_col: np.ndarray
    ) -> jnp.ndarray
        Get the scaled covariance matrix for the specified measurements, and row and column observable names.

    Examples
    --------
    Load theory correlations from HDF5 files in a directory:

    >>> TheoryCorrelations.load('path/to/directory')

    Load theory correlations from a single HDF5 file:

    >>> TheoryCorrelations.load('path/to/file.hdf5')

    Get correlation data for specific row and column observable names:

    >>> data = TheoryCorrelations.get_data(['obs1', 'obs2'], ['obs3', 'obs4'])

    Get scaled covariance matrix for specific measurements and observable names:

    >>> cov_scaled = TheoryCorrelations.get_cov_scaled(
    ...     include_measurements=['meas1', 'meas2'],
    ...     row_names=['obs1', 'obs2'],
    ...     col_names=['obs3', 'obs4'],
    ...     std_th_scaled_row=np.array([[0.1, 0.2], [0.3, 0.4]]),
    ...     std_th_scaled_col=np.array([[0.5, 0.6], [0.7, 0.8]])
    '''

    _correlations: Dict[str, 'TheoryCorrelations'] = {}
    _covariance_scaled: Dict[str, jnp.ndarray] = {}
    _popxf_h5_versions = {'1.0'} # Set of supported versions of the popxf-h5 JSON schema

    def __init__(
        self,
        hash_val: str,
        data: np.ndarray,
        row_names: Dict[str, int],
        col_names: Dict[str, int]
    ) -> None:
        '''
        Initialize an instance of the `TheoryCorrelations` class.

        Parameters
        ----------
        hash_val : str
            The hash value representing the combination of row and column observable names.
        data : np.ndarray
            The data array containing the correlation data.
        row_names : Dict[str, int]
            A dictionary mapping row observable names to their indices.
        col_names : Dict[str, int]
            A dictionary mapping column observable names to their indices.

        Returns
        -------
        None

        Examples
        --------
        Initialize a TheoryCorrelations instance:

        >>> theory_corr = TheoryCorrelations(...)
        '''
        self.hash_val = hash_val
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self._correlations[hash_val] = self

    @classmethod
    def _load_file(cls, path: str) -> None:
        '''
        Load theory correlations from a single HDF5 file.

        Parameters
        ----------
        path : str
            The path to the HDF5 file.

        Returns
        -------
        None

        Examples
        --------
        Load theory correlations from a single HDF5 file:

        >>> TheoryCorrelations._load_file('path/to/file.hdf5')
        '''
        with h5py.File(path, 'r') as f:
            schema_name, schema_version = get_json_schema(dict(f.attrs))
            if schema_name == 'popxf-h5' and schema_version in cls._popxf_h5_versions:
                for hash_val in f:
                    cls.from_hdf5_group(hash_val, f[hash_val])

    @classmethod
    def load(cls, path: str) -> None:
        '''
        Load theory correlations from HDF5 files in the specified path.

        Parameters
        ----------
        path : str
            The path to a directory containing HDF5 files or a single HDF5 file.

        Returns
        -------
        None

        Examples
        --------
        Load theory correlations from HDF5 files in a directory:

        >>> TheoryCorrelations.load('path/to/directory')

        Load theory correlations from a single HDF5 file:

        >>> TheoryCorrelations.load('path/to/file.hdf5')
        '''
        # load all hdf5 files in the directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.hdf5'):
                    cls._load_file(os.path.join(path, file))
        # load single hdf5 file
        else:
            cls._load_file(path)

    @classmethod
    def from_hdf5_group(cls, hash_val: str, hdf5_group: h5py.Group) -> None:
        '''
        Create a `TheoryCorrelations` instance from an HDF5 group.

        Parameters
        ----------
        hash_val : str
            The hash value representing the combination of row and column observable names.
        hdf5_group : h5py.Group
            The HDF5 group containing the correlation data.

        Returns
        -------
        None

        Examples
        --------
        Create a `TheoryCorrelations` instance from an HDF5 group:

        >>> TheoryCorrelations.from_hdf5_group('hash_value', hdf5_group)
        '''
        data = hdf5_group['data']
        data = np.array(data[()], dtype=np.float64) * data.attrs.get('scale', 1.0)
        row_names = {name: i for i, name in enumerate(hdf5_group['row_names'][()].astype(str))}
        col_names = {name: i for i, name in enumerate(hdf5_group['col_names'][()].astype(str))}
        cls(hash_val, data, row_names, col_names)

    @classmethod
    def get_data(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
    ):
        '''
        Get the correlation data for the specified row and column observable names.

        Parameters
        ----------
        row_names : Iterable[str]
            The names of the row observables.
        col_names : Iterable[str]
            The names of the column observables.

        Returns
        -------
        np.ndarray or None
            The correlation data array if found, otherwise None.

        Examples
        --------
        Get correlation data for specific row and column observable names:

        >>> data = TheoryCorrelations.get_data(['obs1', 'obs2'], ['obs3', 'obs4'])
        '''
        hash_val = hash_names(row_names, col_names)
        if hash_val in cls._correlations:
            data = cls._correlations[hash_val].data
        else:
            hash_val = hash_names(col_names, row_names)
            if hash_val in cls._correlations:
                data = np.moveaxis(
                    cls._correlations[hash_val].data,
                    [0,1,2,3], [1,0,3,2]
                )
            else:
                data = None
        return data

    @classmethod
    def get_cov_scaled(
        cls,
        include_measurements: Iterable[str],
        row_names: Iterable[str],
        col_names: Iterable[str],
        std_th_scaled_row: np.ndarray,
        std_th_scaled_col: np.ndarray,
    ):
        '''
        Get the scaled covariance matrix for the specified measurements, and row and column observable names.

        Parameters
        ----------
        include_measurements : Iterable[str]
            The names of the measurements to include.
        row_names : Iterable[str]
            The names of the row observables.
        col_names : Iterable[str]
            The names of the column observables.
        std_th_scaled_row : np.ndarray
            The standard deviations for the row observables.
        std_th_scaled_col : np.ndarray
            The standard deviations for the column observables.

        Returns
        -------
        jnp.ndarray
            The scaled covariance matrix.

        Examples
        --------
        Get scaled covariance matrix for specific measurements and observable names:

        >>> cov_scaled = TheoryCorrelations.get_cov_scaled(
        ...     include_measurements=['meas1', 'meas2'],
        ...     row_names=['obs1', 'obs2'],
        ...     col_names=['obs3', 'obs4'],
        ...     std_th_scaled_row=np.array([[0.1, 0.2], [0.3, 0.4]]),
        ...     std_th_scaled_col=np.array([[0.5, 0.6], [0.7, 0.8]])
        '''
        row_measurements = Measurement.get_measurements(row_names, include_measurements=include_measurements)
        col_measurements = Measurement.get_measurements(col_names, include_measurements=include_measurements)
        hash_val = hash_names(row_measurements, col_measurements, row_names, col_names)
        if hash_val in cls._covariance_scaled:
            cov_scaled = cls._covariance_scaled[hash_val]
        else:
            corr = cls.get_data(row_names, col_names)
            if corr is None:
                raise ValueError(f"Correlation data for {row_names} and {col_names} not found.")
            cov_scaled = corr * np.einsum('ki,lj->ijkl', std_th_scaled_row, std_th_scaled_col)
            cov_scaled = jnp.array(cov_scaled, dtype=jnp.float64)
            cls._covariance_scaled[hash_val] = cov_scaled
        return cov_scaled
