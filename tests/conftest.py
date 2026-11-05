"""Shared fixtures: empty jelli registries, and the synthetic dataset of synthetic_data.py."""

from collections import defaultdict

import pytest

from jelli.core.custom_basis import CustomBasis
from jelli.core.experimental_correlations import ExperimentalCorrelations
from jelli.core.measurement import Measurement
from jelli.core.observable_sector import ObservableSector
from jelli.core.theory_correlations import TheoryCorrelations
from synthetic_data import SyntheticData


@pytest.fixture
def jelli_registries(monkeypatch):
    '''Give the test empty class registries, so that it sees only the data it loads itself.'''
    monkeypatch.setattr(CustomBasis, '_custom_bases', {})
    monkeypatch.setattr(ObservableSector, '_observable_sectors', {})
    monkeypatch.setattr(Measurement, '_measurements', {})
    monkeypatch.setattr(Measurement, '_observable_to_measurements', defaultdict(set))
    monkeypatch.setattr(TheoryCorrelations, '_theory_correlations', {})
    monkeypatch.setattr(ExperimentalCorrelations, '_instances', {'correlations': {}, 'central': {}, 'uncertainties': {}})
    monkeypatch.setattr(ExperimentalCorrelations, '_covariance_scaled', {})
    monkeypatch.setattr(ExperimentalCorrelations, '_observable_names', [])


@pytest.fixture
def synthetic(tmp_path, jelli_registries):
    '''The synthetic dataset, written to `tmp_path` and loaded into empty registries.'''
    return SyntheticData(tmp_path)
