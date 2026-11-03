"""Tests of the code that builds numerical PDFs and normalises them to 1 with
the trapezoidal rule."""
import numpy as np
import pytest

from jelli.core.measurement import Measurement
from jelli.utils.distributions import LOG_ZERO, convert_GeneralGammaDistributionPositive
from jelli.utils.probability import NumericalDistribution


@pytest.fixture(autouse=True)
def no_np_trapz(monkeypatch):
    # NumPy 2.4 removed np.trapz. Remove it here as well, so that these tests
    # fail with any NumPy version if one of these code paths looks it up.
    monkeypatch.delattr(np, "trapz", raising=False)


def integral(y, x):
    """Trapezoidal rule, written out so that it does not depend on the NumPy version."""
    y = np.asarray(y)
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2


def test_numerical_distribution_is_normalised():
    x = np.linspace(0, 2, 21)
    dist = NumericalDistribution(x, 3 * x)
    assert np.isclose(integral(dist.y_norm, x), 1)


def test_general_gamma_with_gaussian_background():
    distribution_type, parameters = convert_GeneralGammaDistributionPositive(
        a=4.0, loc=-1e-6, scale=1e-6, gaussian_standard_deviation=0.5)
    assert distribution_type == "NumericalDistribution"
    x, y = parameters["x"], parameters["y"]
    assert x[0] == 0
    assert np.all(y >= 0)
    assert np.isclose(integral(y, x), 1)
    positive = y > 0
    np.testing.assert_allclose(parameters["log_y"][positive], np.log(y[positive]))


def test_numerical_constraint_is_normalised():
    x = np.linspace(0, 1, 11)
    y = np.where(x < 0.5, 0.0, 2.0)
    constraint = Measurement._define_constraint(
        np.array(["obs"], dtype=object), "NumericalDistribution", x=list(x), y=list(y))
    parameters = constraint["parameters"]
    assert np.isclose(integral(parameters["y"], x), 1)
    assert np.all(parameters["log_y"][y == 0] == LOG_ZERO)
    positive = y > 0
    np.testing.assert_allclose(parameters["log_y"][positive], np.log(parameters["y"][positive]))
