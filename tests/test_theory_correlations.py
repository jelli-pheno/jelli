"""Tests of TheoryCorrelations on a small synthetic popxf-corr file: lookup with rows and columns
in either order, selecting a subset of observables, `scale_factor`, and sector pairs without
correlations. Each result is compared with the old method, which converts the whole tensor to
float64, selects the subset and then scales it with einsum."""

import h5py
import numpy as np
import pytest

from jelli.core.theory_correlations import TheoryCorrelations
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


def test_lookup_cut_and_scale_factor(stored):
    rng = np.random.default_rng(1)
    rows, cols = [0, 2], [1]
    std_row, std_col = _std(rng, N_MONO_ROW, len(rows)), _std(rng, N_MONO_COL, len(cols))
    result = TheoryCorrelations.get_scaled_covariance(ROWS, COLS, rows, cols, std_row, std_col)
    assert result.dtype == np.float64
    assert result.shape == (len(rows), len(cols), N_MONO_ROW, N_MONO_COL)
    np.testing.assert_allclose(
        result, _reference(stored['rows_cols'], SCALE_FACTOR, rows, cols, std_row, std_col), rtol=1e-14
    )


def test_transposed_lookup(stored):
    rng = np.random.default_rng(2)
    rows, cols = [1], [0, 2]  # indices into COLS and into ROWS
    std_row, std_col = _std(rng, N_MONO_COL, len(rows)), _std(rng, N_MONO_ROW, len(cols))
    result = TheoryCorrelations.get_scaled_covariance(COLS, ROWS, rows, cols, std_row, std_col)
    transposed = np.transpose(stored['rows_cols'], (1, 0, 3, 2))
    assert result.shape == (len(rows), len(cols), N_MONO_COL, N_MONO_ROW)
    np.testing.assert_allclose(
        result, _reference(transposed, SCALE_FACTOR, rows, cols, std_row, std_col), rtol=1e-14
    )
    # it is also the transpose of the lookup in the stored order
    direct = TheoryCorrelations.get_scaled_covariance(ROWS, COLS, cols, rows, std_col, std_row)
    np.testing.assert_allclose(result, np.transpose(direct, (1, 0, 3, 2)), rtol=1e-14)


def test_default_scale_factor_and_all_observables(stored):
    rng = np.random.default_rng(3)
    rows = cols = [0, 1, 2]
    std = _std(rng, N_MONO_ROW, 3)
    result = TheoryCorrelations.get_scaled_covariance(ROWS, ROWS, rows, cols, std, std)
    np.testing.assert_allclose(result, _reference(stored['rows_rows'], 1.0, rows, cols, std, std), rtol=1e-14)


def test_missing_pair(stored):
    assert TheoryCorrelations.has_correlations(ROWS, COLS)
    assert TheoryCorrelations.has_correlations(COLS, ROWS)
    assert TheoryCorrelations.has_correlations(ROWS, ROWS)
    assert not TheoryCorrelations.has_correlations(COLS, COLS)
    assert not TheoryCorrelations.has_correlations(['unknown'], COLS)
    result = TheoryCorrelations.get_scaled_covariance(
        COLS, COLS, [0, 1], [0, 1], np.ones((N_MONO_COL, 2)), np.ones((N_MONO_COL, 2))
    )
    assert result is None


def test_nonexistent_returns_none(jelli_registries):
    result = TheoryCorrelations.get_scaled_covariance(
        ['fake_obs'], ['fake_obs'], [0], [0],
        np.ones((1, 1)), np.ones((1, 1))
    )
    assert result is None
