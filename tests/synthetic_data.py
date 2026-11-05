"""A tiny synthetic dataset for the tests, written to a directory (conftest.py has the fixtures).

The dataset uses a custom basis with two real parameters, so no RG evolution is involved. It has
three sectors with Gaussian theory uncertainties, which form one correlated group, and one sector
without theory uncertainties:

- `sector_a`: observables a0, a1, plain polynomials;
- `sector_b`: observables b0, b1, each the ratio of two polynomials (`observable_expressions`);
- `sector_c`: observables c0, c1, plain polynomials;
- `sector_exact`: observable e0, without theory uncertainty.

The measurements constrain a0 and c0 jointly (`meas ac`, a bivariate normal, so the experimental
covariance has an a-c block), a1 (`meas a1`), b0 and b1 (`meas b`) and e0 (`meas e`); c1 is left
unconstrained. The theory correlations of the three Gaussian sectors are blocks of one joint
correlation matrix of all their (observable, monomial) pairs, so that together they are positive
definite. They are stored as int16 with a `scale_factor`, as in real popxf-corr files. The (c, a)
block is stored with rows and columns swapped, so that reading it needs the transposed lookup.
`SyntheticData.correlations` can leave blocks out, set them to zero, write a diagonal block as
uncorrelated (the identity), or write entries with other datasets than `total` or none.
"""

import json

import h5py
import numpy as np

from jelli import GlobalLikelihood
from jelli.core.custom_basis import CustomBasis
from jelli.core.experimental_correlations import ExperimentalCorrelations
from jelli.core.measurement import Measurement
from jelli.core.observable_sector import ObservableSector
from jelli.core.theory_correlations import TheoryCorrelations
from jelli.utils.data_io import hash_names


BASIS = 'jelli test basis'
PAR_LIST = [('p1', 'R'), ('p2', 'R')]
SCALE = 5.0
KEYS = [('', '', 'RR'), ('', 'p1', 'RR'), ('', 'p2', 'RR'), ('p1', 'p1', 'RR'), ('p1', 'p2', 'RR'), ('p2', 'p2', 'RR')]
GAUSSIAN_SECTORS = ['sector_a', 'sector_b', 'sector_c']  # sorted: these are the group-local indices
OBSERVABLES = {'sector_a': ['a0', 'a1'], 'sector_b': ['b0', 'b1'], 'sector_c': ['c0', 'c1'], 'sector_exact': ['e0']}
SM_VALUES = {'sector_a': [1.0, 2.0], 'sector_b': [0.5, 0.25], 'sector_c': [1.5, 3.0], 'sector_exact': [1.0]}
# the polynomials of sector_b: b0 = b0_num / b0_den, b1 = b1_num / b1_den
POLYNOMIALS_B = {'b0_num': 0.5, 'b0_den': 1.0, 'b1_num': 0.5, 'b1_den': 2.0}
MEASUREMENTS = {
    'meas ac': [{'distribution_type': 'MultivariateNormalDistribution', 'observables': ['a0', 'c0'],
                 'central_value': [1.1, 1.4], 'standard_deviation': [0.3, 0.3], 'correlation': [[1.0, 0.4], [0.4, 1.0]]}],
    'meas a1': [{'distribution_type': 'NormalDistribution', 'observables': ['a1'], 'central_value': 2.1, 'standard_deviation': 0.2}],
    'meas b': [{'distribution_type': 'NormalDistribution', 'observables': ['b0'], 'central_value': 0.45, 'standard_deviation': 0.1},
               {'distribution_type': 'NormalDistribution', 'observables': ['b1'], 'central_value': 0.3, 'standard_deviation': 0.1}],
    'meas e': [{'distribution_type': 'NormalDistribution', 'observables': ['e0'], 'central_value': 0.9, 'standard_deviation': 0.2}],
}
INT16_MAX = 32767


def _coefficients(rng, sm_values, scale):
    '''Polynomial coefficients per monomial key: the SM values, and random terms of size `scale`.'''
    return {
        str(key): list(sm_values) if k == 0 else list(scale * rng.normal(size=len(sm_values)))
        for k, key in enumerate(KEYS)
    }


def _uncertainties(rng, sm_values):
    '''Theory uncertainties per monomial key: 5% of the SM values, and random positive terms.'''
    return {
        str(key): list(0.05 * np.abs(sm_values)) if k == 0 else list(0.02 + 0.03 * rng.random(len(sm_values)))
        for k, key in enumerate(KEYS)
    }


def _sector_json(name, rng):
    sm = np.array(SM_VALUES[name])
    metadata = {
        'observable_names': OBSERVABLES[name],
        'parameters': [p for p, _ in PAR_LIST],
        'basis': {'custom': {'name': BASIS}},
        'scale': SCALE,
    }
    data = {'observable_central': _coefficients(rng, sm, 0.3)}
    if name == 'sector_b':
        metadata['polynomial_names'] = list(POLYNOMIALS_B)
        metadata['observable_expressions'] = [
            {'expression': 'num/den', 'variables': {'num': f'{obs}_num', 'den': f'{obs}_den'}}
            for obs in OBSERVABLES[name]
        ]
        data['polynomial_central'] = _coefficients(rng, np.array(list(POLYNOMIALS_B.values())), 0.1)
    if name != 'sector_exact':
        data['observable_uncertainties'] = {'total': _uncertainties(rng, sm)}
    return {'$schema': 'https://json.schemastore.org/popxf-1.0.json', 'metadata': metadata, 'data': data}


class SyntheticData:
    '''Writes the dataset described above to a directory and loads it into jelli.'''

    def __init__(self, path):
        self.path = path
        rng = np.random.default_rng(20261102)
        (path / 'observables').mkdir()
        for name in OBSERVABLES:
            (path / 'observables' / f'{name}.json').write_text(json.dumps(_sector_json(name, rng)))
        (path / 'measurements.json').write_text(json.dumps(
            {'$schema': 'https://json.schemastore.org/pdfxf-1.0.json', **MEASUREMENTS}
        ))
        # joint correlation of all (sector, observable, monomial) triples of the Gaussian sectors
        n = [len(OBSERVABLES[s]) * len(KEYS) for s in GAUSSIAN_SECTORS]
        a = rng.normal(size=(sum(n), 8))
        cov = a @ a.T + 0.5 * np.eye(sum(n))
        d = np.sqrt(np.diag(cov))
        self.joint_correlation = cov / np.outer(d, d)
        self.offsets = np.concatenate([[0], np.cumsum(n)])
        CustomBasis(BASIS, [p for p, _ in PAR_LIST])
        ObservableSector.load(str(path / 'observables'))
        Measurement.load(str(path / 'measurements.json'))
        ExperimentalCorrelations.load()

    def block(self, i, j):
        '''Return the correlation block of Gaussian sectors i and j, with shape (n_obs_i, n_obs_j, n_mono, n_mono).'''
        n_obs_i, n_obs_j = len(OBSERVABLES[GAUSSIAN_SECTORS[i]]), len(OBSERVABLES[GAUSSIAN_SECTORS[j]])
        block = self.joint_correlation[self.offsets[i]:self.offsets[i + 1], self.offsets[j]:self.offsets[j + 1]]
        # rows and columns are ordered (observable, monomial)
        return block.reshape(n_obs_i, len(KEYS), n_obs_j, len(KEYS)).transpose(0, 2, 1, 3)

    def correlations(self, delete=(), zero=(), identity=(), datasets=None, name='correlations.hdf5'):
        '''Write the theory correlations of all sector pairs (i >= j, group-local indices) to a
        popxf-corr file, without the pairs in `delete` and with the pairs in `zero` set to zero.
        The diagonal pairs in `identity` are written as uncorrelated, with the correlations
        δ_mm′ δ_αα′ stored as int16 ones and `scale_factor` 1. `datasets` maps a pair to the
        names of the datasets of its entry, each with the same data (by default `['total']`);
        `[]` writes an entry without any dataset.'''
        path = self.path / name
        with h5py.File(path, 'w') as f:
            f.attrs['$schema'] = 'https://json.schemastore.org/popxf-corr-1.0.json'
            for i in range(len(GAUSSIAN_SECTORS)):
                for j in range(i + 1):
                    if (i, j) in delete:
                        continue
                    rows, cols, block = i, j, self.block(i, j)
                    if (i, j) == (2, 0):  # stored the other way round
                        rows, cols, block = j, i, self.block(j, i)
                    data = np.round(block * INT16_MAX).astype(np.int16)
                    scale_factor = 1 / INT16_MAX
                    if (i, j) in zero:
                        data[...] = 0
                    if (i, j) in identity:
                        assert i == j, 'only a sector with itself can be written as uncorrelated'
                        n_obs = len(OBSERVABLES[GAUSSIAN_SECTORS[i]])
                        data = np.einsum('ij,kl->ijkl', np.eye(n_obs), np.eye(len(KEYS))).astype(np.int16)
                        scale_factor = 1.0
                    row_names = OBSERVABLES[GAUSSIAN_SECTORS[rows]]
                    col_names = OBSERVABLES[GAUSSIAN_SECTORS[cols]]
                    group = f.create_group(hash_names(row_names, col_names))
                    group['row_names'] = np.array(row_names, dtype='S')
                    group['col_names'] = np.array(col_names, dtype='S')
                    entry = group.create_group('correlations')
                    for dataset_name in (datasets or {}).get((i, j), ['total']):
                        dataset = entry.create_dataset(dataset_name, data=data)
                        dataset.attrs['scale_factor'] = scale_factor
        return path

    def likelihood(self, correlations=None, **kwargs):
        '''Build a GlobalLikelihood from the dataset, with the theory correlations from the file
        `correlations` (by default a file with all pairs; `False` loads no file). Other keyword
        arguments are passed on to GlobalLikelihood.'''
        TheoryCorrelations._theory_correlations.clear()
        if correlations is not False:
            TheoryCorrelations.load(str(correlations or self.correlations()))
        return GlobalLikelihood(custom_basis=BASIS, **kwargs)

