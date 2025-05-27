from typing import Dict, Iterable
import h5py
import os
import numpy as np
from jax import numpy as jnp
from jelli.utils.data_io import get_json_schema, hash_observable_names


class TheoryCorrelations:

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
        self.hash_val = hash_val
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self._correlations[hash_val] = self

    @classmethod
    def _load_file(cls, path: str) -> None:
        with h5py.File(path, 'r') as f:
            schema_name, schema_version = get_json_schema(dict(f.attrs))
            if schema_name == 'popxf-h5' and schema_version in cls._popxf_h5_versions:
                for hash_val in f:
                    cls.from_hdf5_group(hash_val, f[hash_val])

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
    def from_hdf5_group(cls, hash_val: str, hdf5_group: h5py.Group) -> None:
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
        hash_val = hash_observable_names(row_names, col_names)
        if hash_val in cls._correlations:
            data = cls._correlations[hash_val].data
        else:
            hash_val = hash_observable_names(col_names, row_names)
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
        row_names: Iterable[str],
        col_names: Iterable[str],
        std_th_scaled_row: np.ndarray,
        std_th_scaled_col: np.ndarray,
    ):
        hash_val = hash_observable_names(row_names, col_names)
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
