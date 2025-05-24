from typing import Dict, List, Optional, Tuple, Iterable
import h5py
import numpy as np
from jax import numpy as jnp
import os
from ..utils.data_io import get_json_schema, hash_observable_names

class ExperimentalCorrelations:

    _instances: Dict[str, Dict[str, 'ExperimentalCorrelations']] = {
        'correlations': {},
        'central': {},
        'uncertainties': {}
    }  # Dictionary to hold instances of ExperimentalCorrelations
    _covariance_scaled: Dict[str, jnp.ndarray] = {}
    _pdfxf_h5_versions = {'1.0'} # Set of supported versions of the pdfxf-h5 JSON schema

    def __init__(
        self,
        hash_val: str,
        data_type: str,
        data: np.ndarray,
        row_names: Dict[str, int],
        col_names: Dict[str, int]
    ) -> None:
        self.hash_val = hash_val
        self.data_type = data_type
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self._instances[data_type][hash_val] = self

    @classmethod
    def _load_file(cls, path: str) -> None:
        with h5py.File(path, 'r') as f:
            schema_name, schema_version = get_json_schema(dict(f.attrs))
            if schema_name == 'pdfxf-h5' and schema_version in cls._pdfxf_h5_versions:
                for hash_val in f['correlations']:
                    cls.from_hdf5_group(hash_val, 'correlations', f['correlations'][hash_val])
                for hash_val in f['central']:
                    cls.from_hdf5_group(hash_val, 'central', f['central'][hash_val])
                for hash_val in f['uncertainties']:
                    cls.from_hdf5_group(hash_val, 'uncertainties', f['uncertainties'][hash_val])

    @classmethod
    def load(cls, path: str) -> None:
        # load all hdf5 files in the directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.hdf5'):
                    cls._load_file(os.path.join(path, file))
        # load single hdf5 file
        else:
            cls._load_file(path)

    @classmethod
    def from_hdf5_group(cls, hash_val: str, data_type: str, hdf5_group: h5py.Group) -> None:
        data = hdf5_group['data']
        data = np.array(data[()], dtype=np.float64) * data.attrs.get('scale', 1.0)
        row_names = {name: i for i, name in enumerate(hdf5_group['row_names'][()].astype(str))}
        col_names = {name: i for i, name in enumerate(hdf5_group.get('col_names', np.array([]))[()].astype(str))}
        cls(hash_val, data_type, data, row_names, col_names)

    @classmethod
    def get_data(
        cls,
        data_type: str,
        row_names: Iterable[str],
        col_names: Optional[Iterable[str]] = []
    ):
        hash_val = hash_observable_names(row_names, col_names)
        if hash_val in cls._instances[data_type]:
            data = cls._instances[data_type][hash_val].data
        else:
            hash_val = hash_observable_names(col_names, row_names)
            if hash_val in cls._instances[data_type]:
                data = cls._instances[data_type][hash_val].data.T
            else:
                data = None
        return data

    @classmethod
    def get_cov_scaled(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
        sigma_exp_scaled_row: np.ndarray,
        sigma_exp_scaled_col: np.ndarray,
    ):
        hash_val = hash_observable_names(row_names, col_names)
        if hash_val in cls._covariance_scaled:
            cov_scaled = cls._covariance_scaled[hash_val]
        else:
            corr = cls.get_data('correlations', row_names, col_names)
            if corr is None:
                raise ValueError(f"Correlation data for {row_names} and {col_names} not found.")
            cov_scaled = corr * np.outer(sigma_exp_scaled_row, sigma_exp_scaled_col)
            cov_scaled = jnp.array(cov_scaled, dtype=jnp.float64)
            cls._covariance_scaled[hash_val] = cov_scaled
        return cov_scaled
