from typing import Dict, Iterable, Optional, Tuple
import h5py
import numpy as np
import os
from jelli.utils.data_io import get_json_schema, hash_names


class TheoryCorrelations:
    '''
    A class to represent theory correlations between polynomial coefficients
    of observable sectors. Correlation arrays are loaded lazily from HDF5 files
    on demand, avoiding persistent memory usage.

    Attributes
    ----------
    _theory_correlations : Dict[str, 'TheoryCorrelations']
        A class attribute mapping hash values to TheoryCorrelations instances.
    _popxf_corr_versions : Set[str]
        A set of supported versions of the popxf-corr JSON schema.

    Methods
    -------
    load(path: str) -> None
        Register theory correlations from HDF5 files in the specified path.
    uncorrelated_scaled_covariance(std_scaled) -> np.ndarray
        Return the scaled covariance block of a sector with itself whose coefficient uncertainties are uncorrelated.
    has_correlations(row_names, col_names) -> bool
        Check whether correlations exist for the specified observable names.
    get_scaled_covariance(row_names, col_names, ...) -> Optional[np.ndarray]
        Load correlations from HDF5, restrict them to the requested observables, and return the scaled covariance.

    Examples
    --------
    Register theory correlations from HDF5 files in a directory:

    >>> TheoryCorrelations.load('path/to/directory')

    Check if correlations exist:

    >>> TheoryCorrelations.has_correlations(['obs1', 'obs2'], ['obs3', 'obs4'])
    True
    '''

    _theory_correlations: Dict[str, 'TheoryCorrelations'] = {}
    _popxf_corr_versions = {'1.0'}

    def __init__(
        self,
        hash_val: str,
        file_path: str,
        group_key: str,
        scale_factor: float,
        row_names: Dict[str, int],
        col_names: Dict[str, int]
    ) -> None:
        self.hash_val = hash_val
        self.file_path = file_path
        self.group_key = group_key
        self.scale_factor = scale_factor
        self.row_names = row_names
        self.col_names = col_names
        self._theory_correlations[hash_val] = self

    @classmethod
    def _find(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
    ) -> Tuple[Optional['TheoryCorrelations'], bool]:
        '''Find the TheoryCorrelations instance for the given observable names.
        Checks both (row, col) and (col, row) hash orderings.

        Returns
        -------
        tc : TheoryCorrelations or None
        transposed : bool
            True if the stored data has row/col axes swapped relative to
            the requested ordering.
        '''
        hash_val = hash_names(row_names, col_names)
        tc = cls._theory_correlations.get(hash_val, None)
        if tc is not None:
            return tc, False
        hash_val = hash_names(col_names, row_names)
        tc = cls._theory_correlations.get(hash_val, None)
        if tc is not None:
            return tc, True
        return None, False

    @classmethod
    def _load_file(cls, path: str) -> None:
        '''
        Register theory correlations from a single HDF5 file (metadata only).

        An entry without any dataset is skipped: the POPxf format defines the uncertainties of a
        key without a dataset as uncorrelated, which is what no entry means. Only the dataset
        `total` can be read; any other dataset raises a `NotImplementedError`.

        The file is registered by its absolute path, so that the correlations, which are read
        when a likelihood is built, are found after a change of the working directory too.

        Parameters
        ----------
        path : str
            The path to the HDF5 file.
        '''
        path = os.path.abspath(path)
        with h5py.File(path, 'r') as f:
            schema_name, schema_version = get_json_schema(dict(f.attrs))
            if schema_name == 'popxf-corr' and schema_version in cls._popxf_corr_versions:
                for hash_val in f:
                    group = f[hash_val]
                    datasets = set(group['correlations'].keys())
                    if not datasets:
                        continue
                    if datasets != {'total'}:
                        raise NotImplementedError(
                            f"Only 'total' correlations are currently supported, but the entry {hash_val} "
                            f"in {path} has the datasets {sorted(datasets)}."
                        )
                    scale_factor = group['correlations']['total'].attrs.get('scale_factor', 1.0)
                    row_names = {name: i for i, name in enumerate(group['row_names'][()].astype(str))}
                    col_names = {name: i for i, name in enumerate(group['col_names'][()].astype(str))}
                    cls(hash_val, path, hash_val, scale_factor, row_names, col_names)

    @classmethod
    def load(cls, path: str) -> None:
        '''
        Register theory correlations from HDF5 files in the specified path.
        Only metadata is stored; correlation arrays are loaded on demand.

        Parameters
        ----------
        path : str
            The path to a directory containing HDF5 files or a single HDF5 file.

        Examples
        --------
        >>> TheoryCorrelations.load('path/to/directory')
        '''
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.hdf5'):
                    cls._load_file(os.path.join(path, file))
        else:
            cls._load_file(path)

    @staticmethod
    def uncorrelated_scaled_covariance(std_scaled: np.ndarray) -> np.ndarray:
        '''Return the scaled covariance block of an observable sector with itself whose
        coefficient uncertainties are uncorrelated: T[m, m, α, α] = std_scaled[α, m]², and zero
        elsewhere.

        The POPxf format defines an entry of a correlation file for each pair of correlated data
        files, so a sector without an entry with itself has the correlations δ_mm′ δ_αα′. The
        block is the one that `get_scaled_covariance` gives for an entry with these correlations.

        Parameters
        ----------
        std_scaled : np.ndarray, shape (n_mono, n_obs)
            Std-scaled theory uncertainties of the sector's observables, as for
            `get_scaled_covariance`. The rows of some monomials give the block of these monomials.

        Returns
        -------
        np.ndarray
            Shape (n_obs, n_obs, n_mono, n_mono), float64.
        '''
        std_scaled = np.asarray(std_scaled, dtype=np.float64)
        n_mono, n_obs = std_scaled.shape
        block = np.zeros((n_obs, n_obs, n_mono, n_mono))
        observables = np.arange(n_obs)[:, np.newaxis]
        monomials = np.arange(n_mono)[np.newaxis, :]
        block[observables, observables, monomials, monomials] = (std_scaled * std_scaled).T
        return block

    @classmethod
    def has_correlations(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
    ) -> bool:
        '''Check whether theory correlations exist for the given observable names
        without loading any data from HDF5.

        Parameters
        ----------
        row_names : Iterable[str]
            The names of the row observables.
        col_names : Iterable[str]
            The names of the column observables.

        Returns
        -------
        bool
            True if correlations exist for this pair (in either ordering).
        '''
        tc, _ = cls._find(row_names, col_names)
        return tc is not None

    @classmethod
    def get_scaled_covariance(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
        row_indices: Iterable[int],
        col_indices: Iterable[int],
        std_scaled_row: np.ndarray,
        std_scaled_col: np.ndarray,
    ) -> Optional[np.ndarray]:
        '''Load the correlation tensor from HDF5, restrict it to the requested
        observables, and return the fully scaled covariance block. Data is kept in native HDF5 dtype (e.g.
        int16) during I/O and converted to float64 only during scaling,
        minimizing peak memory. If either index list is empty, the empty block is returned
        without reading the file.

        Parameters
        ----------
        row_names, col_names : Iterable[str]
            Observable names to look up the correct HDF5 group.
        row_indices, col_indices : Iterable[int]
            Constrained observable indices (subset of full sector).
        std_scaled_row : np.ndarray, shape (n_mono_row, n_row)
            Std-scaled theory uncertainties for row observables.
        std_scaled_col : np.ndarray, shape (n_mono_col, n_col)
            Std-scaled theory uncertainties for col observables.

        Returns
        -------
        np.ndarray or None
            Shape (n_row, n_col, n_mono_row, n_mono_col), float64.
            None if no correlations found for this pair.
        '''
        tc, transposed = cls._find(row_names, col_names)
        if tc is None:
            return None

        row_indices = list(row_indices)
        col_indices = list(col_indices)
        if not row_indices or not col_indices:
            # e.g. a sector without constrained observables in its group: nothing to read
            return np.zeros((len(row_indices), len(col_indices), std_scaled_row.shape[0], std_scaled_col.shape[0]))

        # Load full array once in native dtype (e.g. int16, 1.6 GB for 102×102×276×276).
        # HDF5 datasets are often stored as a single chunk, making per-row reads
        # as expensive as loading the full array. Keeping native dtype avoids the
        # 4× memory expansion of an immediate float64 conversion.
        with h5py.File(tc.file_path, 'r') as f:
            raw = f[tc.group_key]['correlations']['total'][()]

        if transposed:
            # (N_col, N_row, M_col, M_row) → (N_row, N_col, M_row, M_col); view, no copy
            raw = np.moveaxis(raw, [0, 1, 2, 3], [1, 0, 3, 2])

        # Slice to constrained observables in native dtype (cheap copy)
        sliced = raw[np.ix_(row_indices, col_indices)]
        del raw

        # Convert to float64 and scale in-place via broadcasting.
        # result[i,j,k,l] = sliced[i,j,k,l] * scale * std_row[k,i] * std_col[l,j]
        result = sliced.astype(np.float64)
        del sliced
        result *= tc.scale_factor
        result *= std_scaled_row.T[:, np.newaxis, :, np.newaxis]   # (n_row, 1, M_row, 1)
        result *= std_scaled_col.T[np.newaxis, :, np.newaxis, :]   # (1, n_col, 1, M_col)

        return result
