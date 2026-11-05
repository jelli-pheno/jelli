"""Tests of correlated groups on the synthetic dataset of synthetic_data.py: sector pairs without
theory correlations, Gaussian sectors without any constrained observable, and absent correlations
as the POPxf format defines them (WP4.1): a sector without an entry with itself, an entry without
any dataset, and sectors that nothing correlates, which form groups of their own."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jelli.core.experimental_correlations import ExperimentalCorrelations
from jelli.core.observable_sector import ObservableSector
from jelli.core.theory_correlations import TheoryCorrelations
from synthetic_data import GAUSSIAN_SECTORS, OBSERVABLES, PAR_LIST, SCALE


POINTS = [np.zeros(2), np.array([0.3, -0.2]), np.array([-0.25, 0.4])]
UNCONSTRAINED = 'are not included in the likelihood'  # part of the warning about unconstrained sectors


def _assert_same_likelihood(gl, gl_ref, likelihood='global'):
    '''Check that the negative log-likelihood and its gradient agree at POINTS, for cov_order 0, 1 and 2.'''
    for cov_order in [0, 1, 2]:
        nll, data = gl.get_negative_log_likelihood(PAR_LIST, likelihood, cov_order=cov_order)
        nll_ref, data_ref = gl_ref.get_negative_log_likelihood(PAR_LIST, likelihood, cov_order=cov_order)
        for x in POINTS:
            value_ref = nll_ref(x, SCALE, data_ref)
            assert np.isfinite(value_ref)
            np.testing.assert_allclose(nll(x, SCALE, data), value_ref, rtol=1e-12)
            grad_ref = jax.grad(nll_ref)(x, SCALE, data_ref)
            np.testing.assert_allclose(
                jax.grad(nll)(x, SCALE, data), grad_ref,
                rtol=1e-12, atol=1e-12 * np.max(np.abs(grad_ref)),
            )


def _unconstrained_warnings(record):
    return [str(w.message) for w in record if UNCONSTRAINED in str(w.message)]


def test_missing_pair_is_a_none_placeholder(synthetic):
    gl = synthetic.likelihood(synthetic.correlations(delete=[(1, 0)]))
    assert gl.observable_sectors_correlated == [GAUSSIAN_SECTORS]
    blocks = gl.cov_coeff_th_scaled[0]
    assert [len(row) for row in blocks] == [1, 2, 3]
    for i, row in enumerate(blocks):
        for j, block in enumerate(row):
            assert (block is None) == ((i, j) == (1, 0))
    # the blocks hold the constrained observables only: a0 a1 | b0 b1 | c0
    assert [blocks[i][i].shape[0] for i in range(3)] == [2, 2, 1]


def test_missing_pair_equals_zero_pair(synthetic):
    gl_missing = synthetic.likelihood(synthetic.correlations(delete=[(1, 0)], name='missing.hdf5'))
    gl_zeroed = synthetic.likelihood(synthetic.correlations(zero=[(1, 0)], name='zeroed.hdf5'))
    # the zeroed file gives an explicit all-zero tensor, which the None block must reproduce
    block = gl_zeroed.cov_coeff_th_scaled[0][1][0]
    assert block is not None and not np.any(np.asarray(block))
    _assert_same_likelihood(gl_missing, gl_zeroed)
    # _log_likelihood_point contracts the blocks in its own code, so check it as well
    par_array = gl_missing._get_par_array(dict(zip(PAR_LIST, POINTS[1])))
    for cov_order in [0, 1, 2]:
        np.testing.assert_allclose(
            gl_missing._log_likelihood_point(par_array, SCALE, cov_order=cov_order)[5],
            gl_zeroed._log_likelihood_point(par_array, SCALE, cov_order=cov_order)[5],
            rtol=1e-12,
        )


def _theory_blocks(gl, group=0):
    '''The theory covariance blocks of a correlated group, as numpy arrays (None stays None).'''
    return [[None if block is None else np.asarray(block) for block in row] for row in gl.cov_coeff_th_scaled[group]]


def test_missing_self_entry_is_uncorrelated(synthetic):
    # By the POPxf format, a sector without an entry with itself has uncorrelated coefficient
    # uncertainties, which an entry with the correlations δ_mm′ δ_αα′ writes out. The pairs of
    # sector_b with the others are zero, so that it stays in the group and the correlation matrix
    # stays positive definite.
    zero = [(1, 0), (2, 1)]
    gl_missing = synthetic.likelihood(synthetic.correlations(delete=[(1, 1)], zero=zero, name='missing.hdf5'))
    gl_identity = synthetic.likelihood(synthetic.correlations(identity=[(1, 1)], zero=zero, name='identity.hdf5'))
    assert gl_missing.observable_sectors_correlated == [GAUSSIAN_SECTORS]
    block = _theory_blocks(gl_missing)[1][1]
    np.testing.assert_array_equal(block, _theory_blocks(gl_identity)[1][1])
    # nonzero exactly at T[m, m, α, α]
    m, m_col, a, a_col = np.nonzero(block)
    assert np.array_equal(m, m_col) and np.array_equal(a, a_col)
    assert len(m) == block.shape[0] * block.shape[2]
    _assert_same_likelihood(gl_missing, gl_identity)


def test_uncorrelated_block():
    std = np.random.default_rng(3).uniform(0.1, 2.0, size=(4, 3))  # (n_mono, n_obs)
    block = TheoryCorrelations.uncorrelated_scaled_covariance(std)
    assert block.shape == (3, 3, 4, 4) and block.dtype == np.float64
    np.testing.assert_array_equal(block, np.einsum('ij,kl,ki->ijkl', np.eye(3), np.eye(4), std * std))
    # the rows of some monomials give the block of these monomials
    monomials = [0, 2, 3]
    np.testing.assert_array_equal(
        TheoryCorrelations.uncorrelated_scaled_covariance(std[monomials]),
        block[:, :, monomials, :][:, :, :, monomials],
    )


def test_no_correlation_file(synthetic):
    # every sector has uncorrelated coefficient uncertainties, as with an identity entry each
    gl_none = synthetic.likelihood(False)
    gl_identity = synthetic.likelihood(synthetic.correlations(
        identity=[(0, 0), (1, 1), (2, 2)], delete=[(1, 0), (2, 0), (2, 1)],
    ))
    # meas ac correlates a0 and c0; nothing correlates sector_b with the others
    groups = [['sector_a', 'sector_c'], ['sector_b']]
    assert gl_none.observable_sectors_correlated == groups == gl_identity.observable_sectors_correlated
    for g, group in enumerate(groups):
        for row_none, row_identity in zip(_theory_blocks(gl_none, g), _theory_blocks(gl_identity, g)):
            for block_none, block_identity in zip(row_none, row_identity):
                if block_identity is None:
                    assert block_none is None
                else:
                    np.testing.assert_array_equal(block_none, block_identity)
    _assert_same_likelihood(gl_none, gl_identity)


@pytest.mark.parametrize('pair', [(1, 0), (1, 1)])
def test_entry_without_datasets_is_no_entry(synthetic, pair):
    zero = [(1, 0), (2, 1)] if pair == (1, 1) else []  # as in test_missing_self_entry_is_uncorrelated
    gl_empty = synthetic.likelihood(synthetic.correlations(datasets={pair: []}, zero=zero, name='empty.hdf5'))
    # the entry is not registered
    names = [OBSERVABLES[GAUSSIAN_SECTORS[i]] for i in pair]
    assert not TheoryCorrelations.has_correlations(*names)
    gl_deleted = synthetic.likelihood(synthetic.correlations(delete=[pair], zero=zero, name='deleted.hdf5'))
    blocks_empty, blocks_deleted = _theory_blocks(gl_empty), _theory_blocks(gl_deleted)
    if pair == (1, 0):
        assert blocks_empty[1][0] is None and blocks_deleted[1][0] is None
    else:
        np.testing.assert_array_equal(blocks_empty[1][1], blocks_deleted[1][1])
    _assert_same_likelihood(gl_empty, gl_deleted)


@pytest.mark.parametrize('datasets', [['total', 'scale'], ['scale']])
def test_dataset_other_than_total_raises(synthetic, datasets):
    with pytest.raises(NotImplementedError, match=r"\['scale'"):
        synthetic.likelihood(synthetic.correlations(datasets={(1, 0): datasets}))


def test_uncorrelated_sector_forms_its_own_group(synthetic):
    # sector_b has no theory entries with the others, and no measurement correlates it with them
    gl_split = synthetic.likelihood(synthetic.correlations(delete=[(1, 0), (2, 1)], name='split.hdf5'))
    # the same correlations, with the two pairs as zero entries, which link the sectors into one group
    gl_one = synthetic.likelihood(synthetic.correlations(zero=[(1, 0), (2, 1)], name='one.hdf5'))
    assert gl_split.observable_sectors_correlated == [['sector_a', 'sector_c'], ['sector_b']]
    assert gl_one.observable_sectors_correlated == [GAUSSIAN_SECTORS]
    assert gl_split.likelihoods == [('sector_a', 'sector_c'), ('sector_b',), 'sector_exact', 'global']
    _assert_same_likelihood(gl_split, gl_one)
    # the point function adds up the groups in its global log-likelihood
    par_array = gl_split._get_par_array(dict(zip(PAR_LIST, POINTS[1])))
    for cov_order in [0, 1, 2]:
        np.testing.assert_allclose(
            gl_split._log_likelihood_point(par_array, SCALE, cov_order=cov_order)[5][-1],
            gl_one._log_likelihood_point(par_array, SCALE, cov_order=cov_order)[5][-1],
            rtol=1e-12,
        )


def test_no_experimental_block_for_uncorrelated_pair(synthetic, monkeypatch):
    gl = synthetic.likelihood(synthetic.correlations(delete=[(1, 0), (2, 1)]))
    calls = []
    compute = ExperimentalCorrelations.compute
    monkeypatch.setattr(ExperimentalCorrelations, 'compute', lambda *args, **kwargs: calls.append(args) or compute(*args, **kwargs))
    a, b, c = (OBSERVABLES[name] for name in GAUSSIAN_SECTORS)

    def correlations(rows, cols):
        return ExperimentalCorrelations.get_data('correlations', gl.include_measurements, rows, cols)

    assert correlations(a, b) is None and correlations(b, a) is None and correlations(c, b) is None
    block = np.asarray(correlations(a, c))  # meas ac correlates a0 and c0; c1 is unconstrained
    assert block[0, 0] == pytest.approx(0.4) and np.all(np.isnan(block[:, 1]))
    np.testing.assert_array_equal(np.asarray(correlations(c, a)), block.T)
    # computed when gl was built: a pair without correlations is not computed again
    assert calls == []
    ExperimentalCorrelations.load()  # forgets everything, so the next lookup computes
    assert correlations(a, b) is None and len(calls) == 1


def test_no_warning_when_every_sector_is_constrained(synthetic):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        synthetic.likelihood()
    assert not _unconstrained_warnings(record)


@pytest.mark.parametrize('sector, excluded', [
    ('sector_b', ['meas b']),   # observables defined by observable_expressions
    ('sector_c', ['meas ac']),  # plain polynomials; this also leaves a0 unconstrained
])
def test_sector_without_constrained_observable(synthetic, sector, excluded):
    correlations = synthetic.correlations()
    with pytest.warns(UserWarning, match=UNCONSTRAINED) as record:
        gl = synthetic.likelihood(correlations, exclude_measurements=excluded)
    messages = _unconstrained_warnings(record)
    assert len(messages) == 1
    assert sector in messages[0] and all(s not in messages[0] for s in GAUSSIAN_SECTORS if s != sector)

    # the sector stays in the group, but its observables are not included
    assert gl.observable_sectors_correlated == [GAUSSIAN_SECTORS]
    constrained = gl.observables_correlated_constrained[0]
    assert not set(constrained) & set(OBSERVABLES[sector])

    # the n-th prediction must be the prediction of the n-th constrained observable
    x = POINTS[1]
    par_array = gl._get_par_array(dict(zip(PAR_LIST, x)))
    predictions = np.asarray(gl._log_likelihood_point(par_array, SCALE, cov_order=0)[1][0][0])
    expected = {}
    for name in GAUSSIAN_SECTORS:
        observable_sector = ObservableSector.get(name)
        full = observable_sector.prediction(par_array, SCALE, observable_sector.get_prediction_data(None, None))[0]
        expected.update(zip(observable_sector.observable_names, np.asarray(full)))
    np.testing.assert_allclose(predictions, [expected[o] for o in constrained], rtol=1e-14)
    table = gl.parameter_point(dict(zip(PAR_LIST, x)), SCALE).obstable(sort_by='name')
    assert set(table.index) == set(constrained) | set(OBSERVABLES['sector_exact'])
    np.testing.assert_allclose(table.loc[constrained, 'theory'].astype(float), [expected[o] for o in constrained], rtol=1e-14)

    # the likelihood is the same as without this sector
    gl_without = synthetic.likelihood(correlations, exclude_measurements=excluded, exclude_observable_sectors=[sector])
    _assert_same_likelihood(gl, gl_without)


def test_group_without_constrained_observable(synthetic):
    # all Gaussian sectors form a single group; excluding all their measurements leaves the group
    # without any constrained observable
    correlations = synthetic.correlations()
    excluded = ['meas ac', 'meas a1', 'meas b']
    custom = {'mixed': ['a0', 'e0']}
    with pytest.warns(UserWarning, match=UNCONSTRAINED) as record:
        gl = synthetic.likelihood(correlations, exclude_measurements=excluded, custom_likelihoods=custom)
    messages = _unconstrained_warnings(record)
    assert len(messages) == 1 and all(s in messages[0] for s in GAUSSIAN_SECTORS)

    # no group is set up, but the observables are still listed as Gaussian
    assert gl.observable_sectors_correlated == []
    assert sorted(gl.observables_gaussian) == sorted(o for s in GAUSSIAN_SECTORS for o in OBSERVABLES[s])

    gl_exact = synthetic.likelihood(
        correlations, exclude_measurements=excluded, include_observable_sectors=['sector_exact'],
        custom_likelihoods={'mixed': ['e0']},
    )
    _assert_same_likelihood(gl, gl_exact)
    _assert_same_likelihood(gl, gl_exact, likelihood='custom_mixed')


@pytest.mark.parametrize('name', ['sector_a', 'sector_b'])  # plain polynomials, observable_expressions
def test_prediction_function_for_no_observables(synthetic, name):
    sector = ObservableSector.get(name)
    x = jnp.array([0.3, -0.2])
    full, full_data = sector.get_prediction_function(None, None)
    full_predictions, full_monomials = full(x, SCALE, full_data)

    fn, data = sector.get_prediction_function(None, None, observables=[])
    predictions, monomials = jax.jit(fn)(x, SCALE, data)
    assert predictions.shape == (0,)
    np.testing.assert_array_equal(monomials, full_monomials)
    grad = jax.grad(lambda x: jnp.sum(fn(x, SCALE, data)[0]) + jnp.sum(fn(x, SCALE, data)[1]))(x)
    np.testing.assert_allclose(grad, jax.grad(lambda x: jnp.sum(full(x, SCALE, full_data)[1]))(x), rtol=1e-14)

    # predicting a subset gives the matching part of the full prediction
    fn_1, data_1 = sector.get_prediction_function(None, None, observables=sector.observable_names[1:])
    np.testing.assert_allclose(fn_1(x, SCALE, data_1)[0], full_predictions[1:], rtol=1e-14)


@pytest.mark.parametrize('name', ['sector_a', 'sector_b'])  # plain polynomials, observable_expressions
def test_empty_prediction_keeps_batch_axes(synthetic, name):
    # for several parameter points at once, the prediction of no observables has the shape
    # (n_points, 0), as for any other choice of observables (WP4.2)
    sector = ObservableSector.get(name)
    x = jnp.array([[0.3, -0.2], [0.1, 0.0], [0.0, 0.2]])
    full, full_data = sector.get_prediction_function(None, None)
    fn, data = sector.get_prediction_function(None, None, observables=[])
    for function in (fn, jax.jit(fn)):
        predictions, monomials = function(x, SCALE, data)
        assert predictions.shape == (3, 0)
        np.testing.assert_array_equal(monomials, full(x, SCALE, full_data)[1])
