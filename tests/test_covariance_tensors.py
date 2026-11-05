"""Tests of the theory-covariance tensors that the likelihoods contract, on the synthetic dataset of
synthetic_data.py: the par_list cut of the covariance monomials, and the tensors that are built when
a likelihood needs them, shared between likelihoods and freed with the last one that uses them.

The synthetic dataset is in a custom basis, which has no RG evolution. A monomial can then be
nonzero only if each of its factors is the constant or a parameter in par_list, which is the case
of PR #99. The cut under RG mixing is tested in test_par_helpers.py."""

import copy
import gc
import threading
import time

import jax
import numpy as np
import pytest

from jelli import GlobalLikelihood
from jelli.core.observable_sector import ObservableSector
from jelli.core.theory_correlations import TheoryCovarianceSource
from jelli.utils.distributions import cov_coeff_to_cov_obs
from synthetic_data import GAUSSIAN_SECTORS, PAR_LIST, SCALE


POINTS = [np.array([0.3]), np.array([-0.25])]
P1, P2 = [PAR_LIST[0]], [PAR_LIST[1]]
# the sector pairs of the group with theory correlations (all, in the default file)
PAIRS = [(i, j) for i in range(len(GAUSSIAN_SECTORS)) for j in range(i + 1)]


def _uncut_cov_coeff_indices(self, par_indices, cov_order):
    '''The monomials kept without the par_list cut: those of order <= 1 at cov_order 1, all otherwise.'''
    if cov_order == 1:
        return [list(group) for group in self._coeff_indices_linear]
    return [[None] * len(group) for group in self.observable_sectors_correlated]


def _tensors(data):
    '''The theory covariance blocks in the data of a likelihood (the first group; the dataset has one).'''
    return data[4][0][0]


@pytest.fixture
def gl(synthetic):
    return synthetic.likelihood()


@pytest.fixture
def count_builds(monkeypatch):
    '''Count the calls of TheoryCovarianceSource.build, by the monomials they asked for.'''
    calls = []
    build = TheoryCovarianceSource.build
    def counting_build(self, row_monomials=None, col_monomials=None):
        calls.append((row_monomials is None, col_monomials is None))
        return build(self, row_monomials, col_monomials)
    monkeypatch.setattr(TheoryCovarianceSource, 'build', counting_build)
    return calls


@pytest.mark.parametrize('cov_order', [1, 2])
def test_par_list_cut(gl, monkeypatch, cov_order):
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 1.0)  # cut whatever can be cut
    nll_cut, data_cut = gl.get_negative_log_likelihood(P1, 'global', cov_order=cov_order)
    with monkeypatch.context() as m:
        m.setattr(GlobalLikelihood, '_get_cov_coeff_indices', _uncut_cov_coeff_indices)
        nll_uncut, data_uncut = gl.get_negative_log_likelihood(P1, 'global', cov_order=cov_order)
    for x in POINTS:
        value_uncut = nll_uncut(x, SCALE, data_uncut)
        assert np.isfinite(value_uncut)
        np.testing.assert_allclose(nll_cut(x, SCALE, data_cut), value_uncut, rtol=1e-12)
        grad_uncut = jax.grad(nll_uncut)(x, SCALE, data_uncut)
        np.testing.assert_allclose(
            jax.grad(nll_cut)(x, SCALE, data_cut), grad_uncut,
            rtol=1e-12, atol=1e-12 * np.max(np.abs(grad_uncut)),
        )

    # the kept monomials: 1, p1 and p1^2 at cov_order 2, and 1 and p1 at cov_order 1
    expected = [('', '', 'RR'), ('', 'p1', 'RR')] + ([('p1', 'p1', 'RR')] if cov_order == 2 else [])
    par_indices = np.array([gl.parameter_basis_split_re_im[p] for p in P1])
    cut = gl._get_cov_coeff_indices(par_indices, cov_order)
    assert len(cut) == 1 and len(cut[0]) == len(GAUSSIAN_SECTORS)
    for name, indices in zip(GAUSSIAN_SECTORS, cut[0]):
        keys = ObservableSector.get(name).keys_coeff_observable
        assert [keys[k] for k in indices] == expected
    # the blocks of the cut likelihood have the kept monomials on their last two axes
    for i, j in PAIRS:
        assert _tensors(data_cut)[i][j].shape[2:] == (len(expected), len(expected))


def test_par_list_cut_keeps_everything_for_all_parameters(gl):
    par_indices = np.array([gl.parameter_basis_split_re_im[p] for p in PAR_LIST])
    # every monomial can be nonzero, so no sector is cut at cov_order 2
    assert gl._get_cov_coeff_indices(par_indices, 2) == [[None] * len(GAUSSIAN_SECTORS)]
    # at cov_order 1, the monomials of order <= 1
    for s, indices in enumerate(gl._get_cov_coeff_indices(par_indices, 1)[0]):
        np.testing.assert_array_equal(indices, gl._coeff_indices_linear[0][s])


def test_a_cut_keeping_most_monomials_uses_the_full_blocks(gl, monkeypatch):
    # the cut to p1 keeps 3 of the 6 monomials of every sector
    par_indices = np.array([gl.parameter_basis_split_re_im[p] for p in P1])
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 0.5)
    assert all(idx is not None for idx in gl._get_cov_coeff_indices(par_indices, 2)[0])
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 0.4)
    assert gl._get_cov_coeff_indices(par_indices, 2) == [[None] * len(GAUSSIAN_SECTORS)]
    # at cov_order 1 the order-2 monomials must be left out, so the cut is always used
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 0.0)
    assert all(idx is not None for idx in gl._get_cov_coeff_indices(par_indices, 1)[0])

    # the likelihood then contracts the full blocks, and gives the same neg-logL as with the cut
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 0.4)
    nll_full, data_full = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    for i, j in PAIRS:
        assert _tensors(data_full)[i][j] is gl._cov_th_full[(0, i, j)]
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 1.0)
    nll_cut, data_cut = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    for x in POINTS:
        np.testing.assert_allclose(nll_full(x, SCALE, data_full), nll_cut(x, SCALE, data_cut), rtol=1e-12)


def test_setup_and_cov_order_0_hold_no_tensor(gl):
    gc.collect()
    assert gl._cov_th_full == {} and len(gl._cov_th_cuts) == 0 and gl._cov_th_point == {}
    assert gl.constraints_correlated_par_dep_cov[0] is None
    sources = gl._cov_th_sources[0]
    assert all(sources[i][j].nbytes > 0 for i, j in PAIRS)
    _, data = gl.get_negative_log_likelihood(PAR_LIST, 'global', cov_order=0)
    assert len(data) == 8 and data[4][0] is None
    assert gl._cov_th_full == {} and len(gl._cov_th_cuts) == 0


def test_same_cut_is_shared_and_freed_with_its_last_likelihood(gl, monkeypatch):
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 1.0)
    _, data_1 = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    _, data_2 = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    _, data_3 = gl.get_negative_log_likelihood(P2, 'global', cov_order=2)
    for i, j in PAIRS:
        assert _tensors(data_1)[i][j] is _tensors(data_2)[i][j]
        assert _tensors(data_1)[i][j] is not _tensors(data_3)[i][j]
    # count only the blocks still in use: garbage, such as the blocks of the constant monomial that
    # setup builds, can wait for the garbage collector (Python 3.10 in CI)
    gc.collect()
    assert len(gl._cov_th_cuts) == 2 * len(PAIRS)
    del data_1
    gc.collect()
    assert len(gl._cov_th_cuts) == 2 * len(PAIRS)  # data_2 still holds the cut of P1
    del data_2, data_3
    gc.collect()
    assert len(gl._cov_th_cuts) == 0
    assert gl._cov_th_full == {}  # no likelihood needed all monomials


def test_point_builds_the_full_blocks_once(gl, count_builds):
    par_dict = dict(zip(PAR_LIST, [0.3, -0.2]))
    point = gl.parameter_point(par_dict, SCALE, cov_order=2)
    assert count_builds == [(True, True)] * len(PAIRS)
    assert sorted(gl._cov_th_full) == [(0, i, j) for i, j in sorted(PAIRS)]
    assert all(gl._cov_th_sources[0][i][j].nbytes == 0 for i, j in PAIRS)  # never both
    # later points and plots use the kept blocks
    gl.parameter_point(dict(zip(PAR_LIST, [-0.1, 0.25])), SCALE, cov_order=2)
    gl.plot_data_2d(lambda x, y: dict(zip(PAR_LIST, [x, y])), SCALE, -0.1, 0.1, -0.1, 0.1, steps=2, cov_order=2)
    assert len(count_builds) == len(PAIRS)
    # a cut is then taken from the full blocks: the same bits as one built from the source
    _, data = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    assert len(count_builds) == len(PAIRS)
    assert point.log_likelihood_summed is not None


def test_cut_from_full_blocks_equals_cut_from_source(synthetic, monkeypatch):
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 1.0)
    gl_source = synthetic.likelihood()
    _, data_source = gl_source.get_negative_log_likelihood(P1, 'global', cov_order=2)
    gl_full = synthetic.likelihood()
    gl_full.parameter_point({}, SCALE, cov_order=2)  # builds the full blocks
    _, data_full = gl_full.get_negative_log_likelihood(P1, 'global', cov_order=2)
    for i, j in PAIRS:
        np.testing.assert_array_equal(np.asarray(_tensors(data_full)[i][j]), np.asarray(_tensors(data_source)[i][j]))


def test_group_with_cut_and_uncut_sectors(synthetic, monkeypatch):
    '''Only the first sector is cut, to the monomials of p1, so that the blocks between it and the
    other sectors are cut along one axis. The neg-logL and its gradient agree with those without the
    cut, and the blocks taken from the full ones equal those built from the source, bit for bit.'''
    monkeypatch.setattr(GlobalLikelihood, '_max_kept_fraction', 1.0)
    cut = GlobalLikelihood._get_cov_coeff_indices
    def cut_first_sector(self, par_indices, cov_order):
        return [[group[0]] + [None] * (len(group) - 1) for group in cut(self, par_indices, cov_order)]
    gl_source, gl_full = synthetic.likelihood(), synthetic.likelihood()
    gl_full.parameter_point({}, SCALE, cov_order=2)  # builds the full blocks
    with monkeypatch.context() as m:
        m.setattr(GlobalLikelihood, '_get_cov_coeff_indices', cut_first_sector)
        nll, data = gl_source.get_negative_log_likelihood(P1, 'global', cov_order=2)
        _, data_full = gl_full.get_negative_log_likelihood(P1, 'global', cov_order=2)
    with monkeypatch.context() as m:
        m.setattr(GlobalLikelihood, '_get_cov_coeff_indices', _uncut_cov_coeff_indices)
        nll_uncut, data_uncut = gl_source.get_negative_log_likelihood(P1, 'global', cov_order=2)
    n_kept, n_all = 3, 6  # the monomials 1, p1 and p1^2 of the 6
    for i, j in PAIRS:
        assert _tensors(data)[i][j].shape[2:] == (n_all if i else n_kept, n_all if j else n_kept)
        np.testing.assert_array_equal(np.asarray(_tensors(data_full)[i][j]), np.asarray(_tensors(data)[i][j]))
    for x in POINTS:
        np.testing.assert_allclose(nll(x, SCALE, data), nll_uncut(x, SCALE, data_uncut), rtol=1e-12)
        grad_uncut = jax.grad(nll_uncut)(x, SCALE, data_uncut)
        np.testing.assert_allclose(
            jax.grad(nll)(x, SCALE, data), grad_uncut, rtol=1e-12, atol=1e-12 * np.max(np.abs(grad_uncut))
        )


def test_point_at_cov_order_1_keeps_its_linear_blocks(gl, count_builds):
    gl.parameter_point({}, SCALE, cov_order=1)
    gl.parameter_point(dict(zip(PAR_LIST, [0.3, -0.2])), SCALE, cov_order=1)
    assert count_builds == [(False, False)] * len(PAIRS)
    assert gl._cov_th_full == {}
    indices, tensors = gl._cov_th_point[1]
    for i, j in PAIRS:
        assert tensors[0][i][j].shape[2:] == (len(indices[0][i]), len(indices[0][j]))


def test_point_at_cov_order_1_uses_the_full_block_of_a_linear_sector(gl, monkeypatch):
    # a sector with only constant and linear monomials: at cov_order 1 the point function uses its
    # full block, which the point at cov_order 2 has built, rather than a copy of all its monomials
    n_monomials = len(ObservableSector.get(GAUSSIAN_SECTORS[0]).keys_coeff_observable)
    linear = [list(group) for group in gl._coeff_indices_linear]
    linear[0][0] = np.arange(n_monomials)  # as if sector 0 had no monomials of order 2
    monkeypatch.setattr(gl, '_coeff_indices_linear', linear)
    par_dict = dict(zip(PAR_LIST, [0.3, -0.2]))
    gl.parameter_point(par_dict, SCALE, cov_order=2)
    gl.parameter_point(par_dict, SCALE, cov_order=1)
    indices, tensors = gl._cov_th_point[1]
    assert indices[0][0] is None
    assert tensors[0][0][0] is gl._cov_th_full[(0, 0, 0)]


def test_cov_th_scaled_par_indep(gl):
    # the 2D covariance where all parameters vanish, from the full blocks as before WP5, bit for bit
    full = gl._get_cov_th_scaled([[None] * len(GAUSSIAN_SECTORS)])[0]
    constant = []
    for name in GAUSSIAN_SECTORS:
        monomials = np.zeros(len(ObservableSector.get(name).keys_coeff_observable))
        monomials[0] = 1.0
        constant.append(monomials)
    np.testing.assert_array_equal(
        np.asarray(gl.cov_th_scaled_par_indep[0]), np.asarray(cov_coeff_to_cov_obs(constant, full))
    )
    # the 4D tensors of every block are not kept as an attribute any more
    assert not hasattr(gl, 'cov_coeff_th_scaled')


def test_group_that_the_likelihood_does_not_use_builds_no_tensors(synthetic):
    # sector_b, without theory entries with the others, forms its own group. A likelihood of the other
    # group gets no selector for it, so that it is not evaluated and its blocks are not built.
    gl = synthetic.likelihood(synthetic.correlations(delete=[(1, 0), (2, 1)], name='split.hdf5'))
    assert gl.observable_sectors_correlated == [['sector_a', 'sector_c'], ['sector_b']]
    nll_ac, data_ac = gl.get_negative_log_likelihood(PAR_LIST, ('sector_a', 'sector_c'), cov_order=2)
    assert data_ac[7][1] is None and data_ac[4][0][1] is None
    assert gl._cov_th_full and {key[0] for key in gl._cov_th_full} == {0}
    # the likelihoods of the two groups and of the sector without theory uncertainty add up to the
    # global one
    parts = [gl.get_negative_log_likelihood(PAR_LIST, name, cov_order=2) for name in [('sector_b',), 'sector_exact']]
    nll_global, data_global = gl.get_negative_log_likelihood(PAR_LIST, 'global', cov_order=2)
    for x in [np.array([0.3, -0.2]), np.array([-0.25, 0.1])]:
        total = nll_ac(x, SCALE, data_ac) + sum(nll(x, SCALE, data) for nll, data in parts)
        np.testing.assert_allclose(total, nll_global(x, SCALE, data_global), rtol=1e-12)


def test_concurrent_requests_of_a_block(gl, monkeypatch):
    # one thread builds a cut of a block from its correlations while another requests the full
    # block, which releases them: the lock makes the second wait until the first is done
    build = TheoryCovarianceSource.build
    def slow_build(self, row_monomials=None, col_monomials=None):
        if row_monomials is not None:
            time.sleep(0.3)  # the cut pauses before it reads the correlations
        return build(self, row_monomials, col_monomials)
    monkeypatch.setattr(TheoryCovarianceSource, 'build', slow_build)
    monomials = np.array([0, 1])
    blocks, errors = {}, []
    def request(name, rows):
        try:
            blocks[name] = gl._get_cov_th_block(0, 0, 0, rows, rows)
        except Exception as error:
            errors.append(error)
    cut = threading.Thread(target=request, args=('cut', monomials))
    full = threading.Thread(target=request, args=('full', None))
    cut.start()
    time.sleep(0.1)
    full.start()
    cut.join()
    full.join()
    assert errors == []
    np.testing.assert_array_equal(
        np.asarray(blocks['cut']), np.asarray(blocks['full'])[:, :, monomials][:, :, :, monomials]
    )


def test_deepcopy_gets_its_own_lock(gl):
    copied = copy.deepcopy(gl)
    assert isinstance(copied._cov_th_lock, type(gl._cov_th_lock))
    assert copied._cov_th_lock is not gl._cov_th_lock
    nll, data = gl.get_negative_log_likelihood(P1, 'global', cov_order=2)
    nll_copied, data_copied = copied.get_negative_log_likelihood(P1, 'global', cov_order=2)
    for x in POINTS:
        np.testing.assert_array_equal(nll(x, SCALE, data), nll_copied(x, SCALE, data_copied))

