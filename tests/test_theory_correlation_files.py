"""Tests of how TheoryCorrelations reads its HDF5 files, on the synthetic dataset of
synthetic_data.py (WP4.2): a file registered by a relative path is still found after a change of
the working directory, and a block with an empty index list is not read at all."""

import os

import h5py
import jax
import numpy as np
import pytest

from jelli import GlobalLikelihood
from jelli.core.theory_correlations import TheoryCorrelations
from synthetic_data import BASIS, PAR_LIST, SCALE


def test_relative_path_survives_chdir(synthetic, tmp_path, monkeypatch):
    path = synthetic.correlations()
    monkeypatch.chdir(tmp_path)
    TheoryCorrelations._theory_correlations.clear()
    TheoryCorrelations.load(os.path.relpath(path))
    assert all(os.path.isabs(tc.file_path) for tc in TheoryCorrelations._theory_correlations.values())
    # the correlations are read when the likelihood is built, after the change of directory
    (tmp_path / 'elsewhere').mkdir()
    monkeypatch.chdir(tmp_path / 'elsewhere')
    gl = GlobalLikelihood(custom_basis=BASIS)

    gl_ref = synthetic.likelihood(path)
    x = np.array([0.3, -0.2])
    for cov_order in [0, 2]:
        nll, data = gl.get_negative_log_likelihood(PAR_LIST, 'global', cov_order=cov_order)
        nll_ref, data_ref = gl_ref.get_negative_log_likelihood(PAR_LIST, 'global', cov_order=cov_order)
        np.testing.assert_allclose(nll(x, SCALE, data), nll_ref(x, SCALE, data_ref), rtol=1e-12)
        np.testing.assert_allclose(jax.grad(nll)(x, SCALE, data), jax.grad(nll_ref)(x, SCALE, data_ref), rtol=1e-12)


def test_empty_blocks_are_not_read(synthetic, monkeypatch):
    # without `meas b`, sector_b has no constrained observable but stays in the group through its
    # theory entries; its three blocks, (b, a), (b, b) and (c, b), have an empty index list
    TheoryCorrelations._theory_correlations.clear()
    TheoryCorrelations.load(str(synthetic.correlations()))
    opened = []
    open_file = h5py.File
    with monkeypatch.context() as m:
        m.setattr(h5py, 'File', lambda name, *args, **kwargs: opened.append(name) or open_file(name, *args, **kwargs))
        with pytest.warns(UserWarning, match='are not included in the likelihood'):
            gl = GlobalLikelihood(custom_basis=BASIS, exclude_measurements=['meas b'])
    assert gl.observable_sectors_correlated == [['sector_a', 'sector_b', 'sector_c']]
    # only the blocks (a, a), (c, a) and (c, c) are read
    assert len(opened) == 3
