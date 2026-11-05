"""Tests of TheoryCorrelations on a small synthetic popxf-corr file: lookup with rows and columns
in either order, selecting a subset of observables, `scale_factor`, and sector pairs without
correlations. Each block is compared with the old method, which converts the whole tensor to
float64, selects the subset and then scales it with einsum. Further tests check the blocks for
some of the monomials, which `TheoryCovarianceSource.build` computes without a second float64
copy, and the source of a sector without an entry with itself, `UncorrelatedCovarianceSource`."""

import mmap

import h5py
import jax
import numpy as np
import pytest

from jelli.core.theory_correlations import (
    TheoryCorrelations, UncorrelatedCovarianceSource, _MAPPED_BYTES, _aligned_empty,
)
from jelli.utils.data_io import hash_names


ROWS = ['r0', 'r1', 'r2']
COLS = ['c0', 'c1']
N_MONO_ROW, N_MONO_COL = 4, 5
SCALE_FACTOR = 2.0 ** -12


def _write_group(f, row_names, col_names, data, scale_factor=None):
    group = f.create_group(hash_names(row_names, col_names))
    group['row_names'] = np.array(row_names, dtype='S')
    group['col_names'] = np.array(col_names, dtype='S')
    dataset = group.create_group('correlations').create_dataset('total', data=data)
    if scale_factor is not None:
        dataset.attrs['scale_factor'] = scale_factor


@pytest.fixture
def stored(tmp_path, jelli_registries):
    '''Write a file with two blocks, (ROWS, COLS) with a scale_factor and (ROWS, ROWS) without one,
    and load its directory. Returns the stored int16 arrays.'''
    rng = np.random.default_rng(7)
    rows_cols = rng.integers(-2**15 + 1, 2**15, size=(3, 2, N_MONO_ROW, N_MONO_COL), dtype=np.int16)
    rows_rows = rng.integers(-2**15 + 1, 2**15, size=(3, 3, N_MONO_ROW, N_MONO_ROW), dtype=np.int16)
    (tmp_path / 'correlations').mkdir()
    with h5py.File(tmp_path / 'correlations' / 'test.hdf5', 'w') as f:
        f.attrs['$schema'] = 'https://json.schemastore.org/popxf-corr-1.0.json'
        _write_group(f, ROWS, COLS, rows_cols, SCALE_FACTOR)
        _write_group(f, ROWS, ROWS, rows_rows)
    TheoryCorrelations.load(str(tmp_path / 'correlations'))
    return {'rows_cols': rows_cols, 'rows_rows': rows_rows}


def _reference(data, scale_factor, row_indices, col_indices, std_row, std_col):
    '''The old method: convert the whole tensor to float64, select the subset, scale with einsum.'''
    corr = np.array(data, dtype=np.float64) * scale_factor
    corr = corr[np.ix_(row_indices, col_indices)]
    return corr * np.einsum('ki,lj->ijkl', std_row, std_col)


def _std(rng, n_mono, n_obs):
    return rng.uniform(0.1, 2.0, size=(n_mono, n_obs))


def _block(row_names, col_names, row_indices, col_indices, std_row, std_col, **monomials):
    source = TheoryCorrelations.get_covariance_source(row_names, col_names, row_indices, col_indices, std_row, std_col)
    return np.asarray(source.build(**monomials))


def test_lookup_cut_and_scale_factor(stored):
    rng = np.random.default_rng(1)
    rows, cols = [0, 2], [1]
    std_row, std_col = _std(rng, N_MONO_ROW, len(rows)), _std(rng, N_MONO_COL, len(cols))
    result = _block(ROWS, COLS, rows, cols, std_row, std_col)
    assert result.dtype == np.float64
    assert result.shape == (len(rows), len(cols), N_MONO_ROW, N_MONO_COL)
    np.testing.assert_allclose(
        result, _reference(stored['rows_cols'], SCALE_FACTOR, rows, cols, std_row, std_col), rtol=1e-14
    )


def test_transposed_lookup(stored):
    rng = np.random.default_rng(2)
    rows, cols = [1], [0, 2]  # indices into COLS and into ROWS
    std_row, std_col = _std(rng, N_MONO_COL, len(rows)), _std(rng, N_MONO_ROW, len(cols))
    result = _block(COLS, ROWS, rows, cols, std_row, std_col)
    transposed = np.transpose(stored['rows_cols'], (1, 0, 3, 2))
    assert result.shape == (len(rows), len(cols), N_MONO_COL, N_MONO_ROW)
    np.testing.assert_allclose(
        result, _reference(transposed, SCALE_FACTOR, rows, cols, std_row, std_col), rtol=1e-14
    )
    # it is also the transpose of the lookup in the stored order
    direct = _block(ROWS, COLS, cols, rows, std_col, std_row)
    np.testing.assert_allclose(result, np.transpose(direct, (1, 0, 3, 2)), rtol=1e-14)


def test_default_scale_factor_and_all_observables(stored):
    rng = np.random.default_rng(3)
    rows = cols = [0, 1, 2]
    std = _std(rng, N_MONO_ROW, 3)
    result = _block(ROWS, ROWS, rows, cols, std, std)
    np.testing.assert_allclose(result, _reference(stored['rows_rows'], 1.0, rows, cols, std, std), rtol=1e-14)


def test_missing_pair(stored):
    assert TheoryCorrelations.has_correlations(ROWS, COLS)
    assert TheoryCorrelations.has_correlations(COLS, ROWS)
    assert TheoryCorrelations.has_correlations(ROWS, ROWS)
    assert not TheoryCorrelations.has_correlations(COLS, COLS)
    assert not TheoryCorrelations.has_correlations(['unknown'], COLS)
    source = TheoryCorrelations.get_covariance_source(
        COLS, COLS, [0, 1], [0, 1], np.ones((N_MONO_COL, 2)), np.ones((N_MONO_COL, 2))
    )
    assert source is None


def test_nonexistent_returns_none(jelli_registries):
    source = TheoryCorrelations.get_covariance_source(
        ['fake_obs'], ['fake_obs'], [0], [0],
        np.ones((1, 1)), np.ones((1, 1))
    )
    assert source is None


def test_full_block_as_computed_before(stored):
    '''The full block is computed with the operations of the code before WP5, bit for bit.'''
    rng = np.random.default_rng(4)
    rows, cols = [2, 0], [0, 1]
    std_row, std_col = _std(rng, N_MONO_ROW, len(rows)), _std(rng, N_MONO_COL, len(cols))
    expected = stored['rows_cols'][np.ix_(rows, cols)].astype(np.float64)
    expected *= SCALE_FACTOR
    expected *= std_row.T[:, np.newaxis, :, np.newaxis]
    expected *= std_col.T[np.newaxis, :, np.newaxis, :]
    np.testing.assert_array_equal(_block(ROWS, COLS, rows, cols, std_row, std_col), expected)


@pytest.mark.parametrize('row_names, col_names, n_row, n_col', [
    (ROWS, COLS, N_MONO_ROW, N_MONO_COL),
    (COLS, ROWS, N_MONO_COL, N_MONO_ROW),  # the transposed lookup
])
def test_block_for_monomials_is_part_of_full_block(stored, row_names, col_names, n_row, n_col):
    rng = np.random.default_rng(5)
    rows, cols = list(range(len(row_names))), list(range(len(col_names)))
    std_row, std_col = _std(rng, n_row, len(rows)), _std(rng, n_col, len(cols))
    source = TheoryCorrelations.get_covariance_source(row_names, col_names, rows, cols, std_row, std_col)
    full = np.asarray(source.build())
    row_monomials = np.sort(rng.choice(n_row, 2, replace=False))
    col_monomials = np.sort(rng.choice(n_col, 3, replace=False))
    for r, c in [(row_monomials, col_monomials), (row_monomials, None), (None, col_monomials), (np.array([0]), np.array([0]))]:
        block = np.asarray(source.build(row_monomials=r, col_monomials=c))
        expected = full
        if r is not None:
            expected = expected[:, :, r, :]
        if c is not None:
            expected = expected[:, :, :, c]
        np.testing.assert_array_equal(block, expected)


def test_release(stored):
    rng = np.random.default_rng(8)
    std_row, std_col = _std(rng, N_MONO_ROW, 3), _std(rng, N_MONO_COL, 2)
    source = TheoryCorrelations.get_covariance_source(ROWS, COLS, [0, 1, 2], [0, 1], std_row, std_col)
    assert source.nbytes == stored['rows_cols'].nbytes
    source.release()
    assert source.nbytes == 0
    with pytest.raises(RuntimeError, match='released'):
        source.build()


def _is_mapped(array):
    '''Whether the data of `array` is a memory mapping of its own.'''
    while isinstance(array, np.ndarray):
        array = array.base
    if isinstance(array, memoryview):
        array = array.obj
    return isinstance(array, mmap.mmap)


def test_aligned_empty_maps_only_large_arrays():
    '''Small arrays are aligned parts of numpy buffers, so that they take no memory mapping each;
    arrays from _MAPPED_BYTES on are mappings of their own, where the system has them.'''
    small, large = _aligned_empty((3, 4, 5)), _aligned_empty((_MAPPED_BYTES // 8,))
    assert small.ctypes.data % 64 == 0 and large.ctypes.data % 64 == 0
    assert not _is_mapped(small)
    assert _is_mapped(large) == hasattr(mmap, 'MAP_PRIVATE')


@pytest.mark.skipif(jax.default_backend() != 'cpu', reason='JAX takes host buffers over only on the CPU')
def test_build_makes_no_second_float64_copy(stored, monkeypatch):
    '''JAX uses the aligned buffer of `build` without copying it, and copies numpy's usual buffers.'''
    aligned = _aligned_empty((3, 4, 5))
    aligned[...] = 1.0
    assert aligned.ctypes.data % 64 == 0
    assert jax.device_put(aligned, may_alias=True).unsafe_buffer_pointer() == aligned.ctypes.data

    put = jax.device_put
    buffers = []
    def recording_put(x, *args, **kwargs):
        buffers.append(x.ctypes.data)
        return put(x, *args, **kwargs)
    monkeypatch.setattr(jax, 'device_put', recording_put)
    rng = np.random.default_rng(9)
    source = TheoryCorrelations.get_covariance_source(
        ROWS, COLS, [0, 1, 2], [0, 1], _std(rng, N_MONO_ROW, 3), _std(rng, N_MONO_COL, 2)
    )
    block = source.build()
    assert block.dtype == np.float64
    assert len(buffers) == 1 and block.unsafe_buffer_pointer() == buffers[0]
    # also the block of a sector without an entry with itself
    buffers.clear()
    block = UncorrelatedCovarianceSource(_std(rng, N_MONO_ROW, 3)).build()
    assert len(buffers) == 1 and block.unsafe_buffer_pointer() == buffers[0]


def test_uncorrelated_source():
    '''The source of a sector without an entry with itself keeps only the scaled uncertainties and
    builds T[m, m, α, α] = σ̂_mα², zero elsewhere, for all or some of the monomials.'''
    std = _std(np.random.default_rng(11), N_MONO_ROW, 3)
    source = UncorrelatedCovarianceSource(std)
    assert source.shape == (3, 3, N_MONO_ROW, N_MONO_ROW) and source.nbytes == 0
    full = np.asarray(source.build())
    np.testing.assert_array_equal(full, TheoryCorrelations.uncorrelated_scaled_covariance(std))
    monomials = np.array([0, 2, 3])
    np.testing.assert_array_equal(source.build(monomials, monomials), full[:, :, monomials][:, :, :, monomials])
    source.release()  # nothing to drop
    np.testing.assert_array_equal(source.build(), full)
    with pytest.raises(ValueError, match='same monomials'):
        source.build(monomials, None)


def test_block_filled_in_threads_equals_block_filled_in_one(stored, monkeypatch):
    '''Large blocks are filled in threads, one row of observables per task; small ones are not.'''
    import jelli.core.theory_correlations as theory_correlations
    rng = np.random.default_rng(10)
    rows, cols = [0, 1, 2], [0, 1]
    std_row, std_col = _std(rng, N_MONO_ROW, len(rows)), _std(rng, N_MONO_COL, len(cols))
    source = TheoryCorrelations.get_covariance_source(ROWS, COLS, rows, cols, std_row, std_col)
    row_monomials, col_monomials = np.array([0, 3]), np.array([1, 2, 4])
    one = [np.asarray(source.build()), np.asarray(source.build(row_monomials, col_monomials))]
    monkeypatch.setattr(theory_correlations, '_THREADED_BYTES', 0)
    monkeypatch.setattr(theory_correlations, '_THREADS', 3)
    threads = [np.asarray(source.build()), np.asarray(source.build(row_monomials, col_monomials))]
    for a, b in zip(one, threads):
        np.testing.assert_array_equal(a, b)
