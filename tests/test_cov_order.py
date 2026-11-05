"""Tests of `cov_order` on the synthetic dataset of synthetic_data.py. Every method taking it
accepts only non-negative integers; bools are rejected, since `True == 1` would otherwise silently
select `cov_order=1`. And at each order, the neg-logL and the parameter point agree with a
reference that computes the theory covariance without jelli's code."""

import numpy as np
import pytest

from jelli.core.compiled_likelihood import CompiledLikelihood
from jelli.core.global_likelihood_point import GlobalLikelihoodPoint
from jelli.core.observable_sector import ObservableSector
from synthetic_data import GAUSSIAN_SECTORS, MEASUREMENTS, PAR_LIST, SCALE


def _par_fct(x, y):
    return {('p1', 'R'): x, ('p2', 'R'): y}


# each passes cov_order as a positional argument where the signature allows it
ENTRY_POINTS = {
    'get_negative_log_likelihood': lambda gl, c: gl.get_negative_log_likelihood(PAR_LIST, 'global', c),
    'get_compiled_likelihood': lambda gl, c: gl.get_compiled_likelihood(PAR_LIST, 'global', c),
    'parameter_point': lambda gl, c: gl.parameter_point({}, SCALE, cov_order=c),
    'plot_data_2d': lambda gl, c: gl.plot_data_2d(_par_fct, SCALE, -0.1, 0.1, -0.1, 0.1, False, False, 2, c),
    'CompiledLikelihood': lambda gl, c: CompiledLikelihood(gl, PAR_LIST, 'global', c),
    'GlobalLikelihoodPoint': lambda gl, c: GlobalLikelihoodPoint(gl, gl._get_par_array({}), SCALE, c),
}


@pytest.fixture
def gl(synthetic):
    return synthetic.likelihood()


@pytest.mark.parametrize('value', [True, False])
@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_bool_raises(gl, name, value):
    with pytest.raises(TypeError, match='use `cov_order=2` for a fully parameter-dependent'):
        ENTRY_POINTS[name](gl, value)


@pytest.mark.parametrize('value, error', [(-1, ValueError), (1.5, TypeError), ('2', TypeError), (None, TypeError)])
@pytest.mark.parametrize('name', ['get_negative_log_likelihood', 'GlobalLikelihoodPoint'])
def test_bad_value_raises(gl, name, value, error):
    with pytest.raises(error, match='cov_order'):
        ENTRY_POINTS[name](gl, value)


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_orders_build(gl, name):
    for cov_order in [0, 1, 2, np.int64(2)]:
        ENTRY_POINTS[name](gl, cov_order)


def test_compiled_likelihood_cache(gl):
    compiled = {c: gl.get_compiled_likelihood(PAR_LIST, 'global', c) for c in [0, 1, 2]}
    assert len({id(c) for c in compiled.values()}) == 3
    assert all(compiled[c].cov_order == c for c in compiled)
    assert gl.get_compiled_likelihood(PAR_LIST, 'global', np.int64(1)) is compiled[1]
    assert type(gl.parameter_point({}, SCALE, cov_order=np.int64(2)).cov_order) is int


def _experimental_mean_and_covariance(observables):
    '''Mean and covariance of `observables` from the measurements of the synthetic dataset, in
    which every observable is constrained by one measurement.'''
    mean, cov = {}, {}
    for constraints in MEASUREMENTS.values():
        for constraint in constraints:
            names = constraint['observables']
            if constraint['distribution_type'] == 'NormalDistribution':
                mean[names[0]] = constraint['central_value']
                cov[names[0], names[0]] = constraint['standard_deviation'] ** 2
            else:  # MultivariateNormalDistribution
                std, corr = constraint['standard_deviation'], constraint['correlation']
                for i, name_i in enumerate(names):
                    mean[name_i] = constraint['central_value'][i]
                    for j, name_j in enumerate(names):
                        cov[name_i, name_j] = corr[i][j] * std[i] * std[j]
    return (
        np.array([mean[o] for o in observables]),
        np.array([[cov.get((o, p), 0.0) for p in observables] for o in observables]),
    )


def _reference_negative_log_likelihood(gl, x, cov_order):
    '''The neg-logL at the point `x`, with the theory covariance computed here instead of by jelli.
    Without any correlation file every coefficient uncertainty is uncorrelated, so the theory
    covariance is diagonal: for each observable, the sum of (V σ)² over the monomials V of order at
    most `cov_order`, where σ is the uncertainty of the monomial's coefficient. As in jelli, the
    neg-logL of a group is half its χ², without the normalization of the Gaussian.'''
    values = {name: value for (name, _), value in zip(PAR_LIST, x)}
    par_array = gl._get_par_array(dict(zip(PAR_LIST, x)))
    prediction, variance = {}, {}
    for name in GAUSSIAN_SECTORS:
        sector = ObservableSector.get(name)
        keys = sector.keys_coeff_observable
        monomials = np.array([(values[w1] if w1 else 1.0) * (values[w2] if w2 else 1.0) for w1, w2, _ in keys])
        kept = np.array([(w1 != '') + (w2 != '') <= cov_order for w1, w2, _ in keys])
        terms = (kept * monomials)[:, np.newaxis] * np.asarray(sector.observable_uncertainties)
        variance.update(zip(sector.observable_names, np.sum(terms ** 2, axis=0)))
        predictions = sector.prediction(par_array, SCALE, sector.get_prediction_data(None, None))[0]
        prediction.update(zip(sector.observable_names, np.asarray(predictions)))
    value = 0.0
    for observables in gl.observables_correlated_constrained:
        mean, cov_exp = _experimental_mean_and_covariance(observables)
        cov = cov_exp + np.diag([variance[o] for o in observables])
        residual = np.array([prediction[o] for o in observables]) - mean
        value += 0.5 * residual @ np.linalg.solve(cov, residual)
    return value


def test_orders_against_reference(synthetic):
    # without any correlation file, the theory covariance is known without jelli's code (WP4.2)
    gl = synthetic.likelihood(False, exclude_observable_sectors=['sector_exact'])
    assert gl.observables_correlated_constrained == [['a0', 'a1', 'c0'], ['b0', 'b1']]
    x = np.array([0.3, -0.2])
    reference = {c: _reference_negative_log_likelihood(gl, x, c) for c in [0, 1, 2]}
    reference_origin = _reference_negative_log_likelihood(gl, np.zeros(2), 0)
    # the orders differ far beyond the tolerance, so that no order can pass for another
    assert min(abs(reference[0] - reference[1]), abs(reference[1] - reference[2])) > 1e-6 * abs(reference[2])
    for cov_order in [0, 1, 2]:
        nll, data = gl.get_negative_log_likelihood(PAR_LIST, 'global', cov_order=cov_order)
        np.testing.assert_allclose(nll(x, SCALE, data), reference[cov_order], rtol=1e-12)
        # the parameter point gives the log-likelihood relative to its value where all parameters vanish
        point = gl.parameter_point(dict(zip(PAR_LIST, x)), SCALE, cov_order=cov_order)
        np.testing.assert_allclose(-point.log_likelihood_global(), reference[cov_order] - reference_origin, rtol=1e-12)
