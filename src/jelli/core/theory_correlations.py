from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional, Tuple
import h5py
import jax
import mmap
import numpy as np
import os
from jelli.utils.data_io import get_json_schema, hash_names


# blocks larger than this are filled in threads, one row of observables per task
_THREADED_BYTES = 64 * 2**20
_THREADS = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
# arrays from this size on get a memory mapping of their own (see `_aligned_empty`); numpy gives its
# own arrays from this size on the huge-page advice
_MAPPED_BYTES = 4 * 2**20


def _aligned_empty(shape):
    '''
    Return an uninitialized float64 array whose data is 64-byte aligned.

    On the CPU, `jax.device_put(array, may_alias=True)` uses such an array without copying it,
    while it copies an array with numpy's usual 16-byte alignment. An array of `_MAPPED_BYTES` or
    more is a private anonymous mapping, as malloc uses for large arrays, without the huge-page
    advice that numpy gives its own large arrays: with transparent huge pages in 'madvise' mode,
    that advice made the first write to a 4.4 GB array take 9.5-12 s instead of 1.0-1.4 s (WP5). A
    smaller array is an aligned part of a numpy buffer, which gets no such advice; a mapping for
    each would count against the number of mappings that a process may have (`vm.max_map_count`,
    65530 by default on Linux).
    '''
    size = int(np.prod(shape))
    if size * 8 >= _MAPPED_BYTES and hasattr(mmap, 'MAP_PRIVATE'):
        buffer = mmap.mmap(-1, size * 8, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        return np.frombuffer(buffer, dtype=np.float64, count=size).reshape(shape)
    # small arrays, and without private mappings (Windows): a 64-byte-aligned part of a numpy buffer
    buffer = np.empty(size * 8 + 64, dtype=np.uint8)
    offset = -buffer.ctypes.data % 64
    return buffer[offset:offset + size * 8].view(np.float64).reshape(shape)


def _fill_block(block, stored, row_monomials=None, col_monomials=None, scale_factor=None,
                std_row=None, std_col=None):
    '''
    Fill the float64 array `block` of shape `(n_row, n_col, n_mono_row, n_mono_col)` from `stored`.

    `block[a, b]` is the part of `stored[a, b]` for the given monomials, converted to float64. If
    `scale_factor` is given, it is then multiplied by `scale_factor`, by the row's scaled
    uncertainties `std_row[:, a]` and by the column's `std_col[:, b]`, in this order, which are
    already restricted to the monomials. The block is filled one observable pair at a time, so that
    the work stays in the cache, and for large blocks in threads, one row of observables per task.
    '''
    rows = None if row_monomials is None else np.asarray(row_monomials)
    cols = None if col_monomials is None else np.asarray(col_monomials)
    n_col = block.shape[1]

    def fill_row(a):
        for b in range(n_col):
            part = stored[a, b]
            if rows is not None and cols is not None:
                part = part[rows[:, np.newaxis], cols[np.newaxis, :]]
            elif rows is not None:
                part = part[rows, :]
            elif cols is not None:
                part = part[:, cols]
            out = block[a, b]
            out[...] = part
            if scale_factor is not None:
                out *= scale_factor
                out *= std_row[:, a, np.newaxis]
                out *= std_col[np.newaxis, :, b]

    n_threads = min(_THREADS, block.shape[0])
    if block.nbytes > _THREADED_BYTES and n_threads > 1:
        with ThreadPoolExecutor(n_threads) as pool:
            list(pool.map(fill_row, range(block.shape[0])))
    else:
        for a in range(block.shape[0]):
            fill_row(a)


def cut_of_full_block(full, row_monomials, col_monomials):
    '''
    Return the block for the given monomials (None: all of an axis), taken from a full float64
    block, as a new JAX array.

    On the CPU, the full block is read through a numpy view, so that no gather has to be compiled
    for the shape of the block; on other devices, it is a JAX gather.
    '''
    if not all(device.platform == 'cpu' for device in full.devices()):
        return select_monomials(full, row_monomials, col_monomials)
    view = np.asarray(full)
    n_rows = view.shape[2] if row_monomials is None else len(row_monomials)
    n_cols = view.shape[3] if col_monomials is None else len(col_monomials)
    block = _aligned_empty((view.shape[0], view.shape[1], n_rows, n_cols))
    _fill_block(block, view, row_monomials, col_monomials)
    return jax.device_put(block, may_alias=True)


def select_monomials(block, row_monomials, col_monomials):
    '''
    Restrict the monomial axes (the last two) of a theory covariance block, in one gather.

    Parameters
    ----------
    block : np.ndarray or jnp.array
        Array of shape `(n_row, n_col, n_mono_row, n_mono_col)`.
    row_monomials, col_monomials : np.ndarray or None
        Indices of the monomials to keep on each of the two axes. None keeps all of them.

    Returns
    -------
    np.ndarray or jnp.array
        `block` itself if both are None, otherwise a new array.
    '''
    if row_monomials is None and col_monomials is None:
        return block
    if row_monomials is None:
        return block[:, :, :, np.asarray(col_monomials)]
    if col_monomials is None:
        return block[:, :, np.asarray(row_monomials), :]
    return block[:, :, np.asarray(row_monomials)[:, np.newaxis], np.asarray(col_monomials)[np.newaxis, :]]


class TheoryCovarianceSource:
    '''
    The theory covariance of two observable sectors in compact form. From it, `build` computes the
    4D block of the covariance for the monomials that a likelihood needs.

    The block has the shape `(n_row, n_col, n_mono_row, n_mono_col)` and the entries
    `correlation[i, j, k, l] * scale_factor * std_scaled_row[k, i] * std_scaled_col[l, j]`, where
    i and j run over the constrained observables of the two sectors. The source keeps the
    correlations of these observables in the dtype of the popxf-corr file. For int16, the dtype of
    real files, they take a quarter of the memory of the float64 block.

    Attributes
    ----------
    shape : tuple
        Shape of the full block.
    scale_factor : float
        Factor that converts the stored correlations to correlations.
    std_scaled_row, std_scaled_col : np.ndarray
        Scaled theory uncertainties of the row and the column observables, of shapes
        `(n_mono_row, n_row)` and `(n_mono_col, n_col)`.
    nbytes : int
        Memory of the stored correlations, 0 once they are released.

    Methods
    -------
    build(row_monomials=None, col_monomials=None) -> jax.Array
        Compute the float64 block for the given monomials.
    release() -> None
        Drop the stored correlations.
    '''

    def __init__(self, correlations, scale_factor, std_scaled_row, std_scaled_col):
        self._correlations = correlations
        self.shape = correlations.shape
        self.scale_factor = scale_factor
        self.std_scaled_row = std_scaled_row
        self.std_scaled_col = std_scaled_col

    @property
    def nbytes(self):
        return 0 if self._correlations is None else self._correlations.nbytes

    def release(self):
        '''Drop the stored correlations, once the full block is kept elsewhere. `build` then raises.'''
        self._correlations = None

    def build(self, row_monomials=None, col_monomials=None):
        '''
        Compute the float64 block for the given monomials.

        The entries are computed with the same operations in the same order as the full block was
        computed before: the correlation is converted to float64 and multiplied by `scale_factor`,
        then by the row's and then by the column's scaled uncertainty. A block for some of the
        monomials therefore equals the same part of the full block, bit for bit. It is computed in a
        64-byte-aligned buffer, which JAX then uses without a copy, so that no two float64 copies of
        the block exist at any time.

        Parameters
        ----------
        row_monomials, col_monomials : np.ndarray or None
            Indices of the monomials of the row and of the column sector. None keeps all of them.

        Returns
        -------
        jax.Array
            Array of shape `(n_row, n_col, len(row_monomials), len(col_monomials))`.
        '''
        correlations = self._correlations  # read once: a release during the build does not affect it
        if correlations is None:
            raise RuntimeError('The correlations of this source have been released.')
        std_row = self.std_scaled_row if row_monomials is None else self.std_scaled_row[np.asarray(row_monomials)]
        std_col = self.std_scaled_col if col_monomials is None else self.std_scaled_col[np.asarray(col_monomials)]
        block = _aligned_empty((self.shape[0], self.shape[1], std_row.shape[0], std_col.shape[0]))
        _fill_block(block, correlations, row_monomials, col_monomials, self.scale_factor, std_row, std_col)
        return jax.device_put(block, may_alias=True)


class UncorrelatedCovarianceSource:
    '''
    The theory covariance of an observable sector with itself whose coefficient uncertainties are
    uncorrelated, as the POPxf format defines it for a sector without an entry with itself in the
    correlation files: T[m, m, α, α] = σ̂_mα², and zero elsewhere. The source keeps only the scaled
    uncertainties, and `build` computes the block with
    `TheoryCorrelations.uncorrelated_scaled_covariance`.

    `GlobalLikelihood` uses it like a `TheoryCovarianceSource`: it has the same `shape`, `nbytes`,
    `build` and `release`. `nbytes` is 0, since it stores no correlations, and `release` does
    nothing.

    Attributes
    ----------
    shape : tuple
        Shape of the full block, `(n_obs, n_obs, n_mono, n_mono)`.
    std_scaled : np.ndarray
        Scaled theory uncertainties of the observables, of shape `(n_mono, n_obs)`.
    nbytes : int
        Memory of the stored correlations: 0.
    '''

    nbytes = 0

    def __init__(self, std_scaled):
        self.std_scaled = std_scaled
        n_mono, n_obs = std_scaled.shape
        self.shape = (n_obs, n_obs, n_mono, n_mono)

    def release(self):
        '''Nothing to drop: the block is computed from the scaled uncertainties alone.'''

    def build(self, row_monomials=None, col_monomials=None):
        '''
        Compute the float64 block for the given monomials, which are the same for the rows and the
        columns, as for every block of a sector with itself. It equals the same part of the full
        block, bit for bit, and JAX uses its buffer without a copy.

        Parameters
        ----------
        row_monomials, col_monomials : np.ndarray or None
            Indices of the monomials, the same on both axes. None keeps all of them.

        Returns
        -------
        jax.Array
            Array of shape `(n_obs, n_obs, n_kept, n_kept)`.
        '''
        same = (row_monomials is None and col_monomials is None) or (
            row_monomials is not None and col_monomials is not None
            and np.array_equal(row_monomials, col_monomials)
        )
        if not same:
            raise ValueError('The block of a sector with itself is built for the same monomials on both axes.')
        std = self.std_scaled if row_monomials is None else self.std_scaled[np.asarray(row_monomials)]
        return jax.device_put(TheoryCorrelations.uncorrelated_scaled_covariance(std), may_alias=True)


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
    get_covariance_source(row_names, col_names, ...) -> Optional[TheoryCovarianceSource]
        Load the correlations of the requested observables, from which the scaled covariance block is built.

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
        block is the one that `TheoryCovarianceSource.build` gives for an entry with these
        correlations. It is computed in a 64-byte-aligned buffer, which JAX uses without a copy.

        Parameters
        ----------
        std_scaled : np.ndarray, shape (n_mono, n_obs)
            Std-scaled theory uncertainties of the sector's observables, as for
            `get_covariance_source`. The rows of some monomials give the block of these monomials.

        Returns
        -------
        np.ndarray
            Shape (n_obs, n_obs, n_mono, n_mono), float64.
        '''
        std_scaled = np.asarray(std_scaled, dtype=np.float64)
        n_mono, n_obs = std_scaled.shape
        block = _aligned_empty((n_obs, n_obs, n_mono, n_mono))
        block[...] = 0.0
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
    def get_covariance_source(
        cls,
        row_names: Iterable[str],
        col_names: Iterable[str],
        row_indices: Iterable[int],
        col_indices: Iterable[int],
        std_scaled_row: np.ndarray,
        std_scaled_col: np.ndarray,
    ) -> Optional[TheoryCovarianceSource]:
        '''
        Return the source of the scaled theory covariance block of two observable sectors,
        restricted to the requested observables, or None if no correlations are stored for them.

        The correlations are read in their stored dtype (int16 in real files) and restricted to
        the requested observables before anything is converted to float64; `build` of the source
        then computes the float64 block for chosen monomials. If either index list is empty, the
        file is not read, and the source holds the empty block.

        Parameters
        ----------
        row_names, col_names : Iterable[str]
            Observable names of the two sectors, which identify the HDF5 group.
        row_indices, col_indices : Iterable[int]
            Indices of the requested (constrained) observables of each sector.
        std_scaled_row : np.ndarray, shape (n_mono_row, n_row)
            Scaled theory uncertainties of the requested row observables.
        std_scaled_col : np.ndarray, shape (n_mono_col, n_col)
            Scaled theory uncertainties of the requested column observables.

        Returns
        -------
        TheoryCovarianceSource or None
            The source of the block of shape (n_row, n_col, n_mono_row, n_mono_col), or None if no
            correlations are stored for this pair.
        '''
        tc, transposed = cls._find(row_names, col_names)
        if tc is None:
            return None
        row_indices = list(row_indices)
        col_indices = list(col_indices)
        if not row_indices or not col_indices:
            # e.g. a sector without constrained observables in its group: nothing to read
            empty = np.zeros((len(row_indices), len(col_indices), std_scaled_row.shape[0], std_scaled_col.shape[0]))
            return TheoryCovarianceSource(empty, tc.scale_factor, std_scaled_row, std_scaled_col)
        # The whole array is read in its stored dtype: HDF5 datasets are often stored as one chunk,
        # which makes reading a part of them as expensive as reading all of it.
        with h5py.File(tc.file_path, 'r') as f:
            raw = f[tc.group_key]['correlations']['total'][()]
        if transposed:
            # (N_col, N_row, M_col, M_row) -> (N_row, N_col, M_row, M_col); a view, no copy
            raw = np.moveaxis(raw, [0, 1, 2, 3], [1, 0, 3, 2])
        correlations = raw[np.ix_(row_indices, col_indices)]  # in the stored dtype
        del raw
        return TheoryCovarianceSource(correlations, tc.scale_factor, std_scaled_row, std_scaled_col)
