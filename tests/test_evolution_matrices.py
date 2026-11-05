"""The RG evolution matrices of an observable sector in rgevolve mode.

`ObservableSector` builds each evolution matrix with one call of `rgevolve.tools.run_and_match`,
with `sector_out=None` and the lists of input and output Wilson coefficients. The tests compare the
matrices with a reference built sector by sector from `run_and_match(..., sector_out=s)`. They also
check the predictions and the neg-logL of a likelihood in the SMEFT that uses the matrices.

The observable sector is synthetic: two observables in the WET flavio basis at 4.8 GeV, with
parameters in three rgevolve sectors. `sbemu` and `sbmue` both match from the SMEFT sector `mue`,
and `etauemu` matches from the SMEFT sector `etauemu`. So in the SMEFT matrices two sectors share
input columns, and the blocks between sectors that do not match are zero. The tests need the
rgevolve WET and SMEFT bases, which the `rgevolve` dependency installs.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rgevolve.tools import get_scales, get_wc_basis, reference_scale, run_and_match

from jelli import GlobalLikelihood
from jelli.core.experimental_correlations import ExperimentalCorrelations
from jelli.core.measurement import Measurement
from jelli.core.observable_sector import ObservableSector


NAME = 'wet_sector'
EFT, BASIS, SCALE = 'WET', 'flavio', 4.8
SECTORS = ['sbemu', 'sbmue', 'etauemu']
PARAMETERS = ['C9_bsmue', 'C10_bsmue', 'C9_bsemu', 'CVLL_muetaue']
SM_VALUES = [1.0, 2.0]
# measurements of the two observables: central value and standard deviation
MEASUREMENTS = {'o0': (1.1, 0.1), 'o1': (1.9, 0.2)}
# the (eft, basis) pairs with evolution matrices, the input sectors of each EFT in the order of
# the columns, and the input sector of each sector of the observable sector
EVOLUTIONS = [('SMEFT', 'Warsaw'), ('SMEFT', 'Warsaw up'), ('WET', 'JMS'), ('WET', 'flavio')]
SECTORS_IN = {'SMEFT': ['etauemu', 'mue'], 'WET': ['etauemu', 'sbemu', 'sbmue']}
SECTOR_IN = {
    'SMEFT': {'etauemu': 'etauemu', 'sbemu': 'mue', 'sbmue': 'mue'},
    'WET': {'etauemu': 'etauemu', 'sbemu': 'sbemu', 'sbmue': 'sbmue'},
}
SMEFT_SCALE = 1000.0  # one of rgevolve's SMEFT scales, so that no interpolation is involved


def _load_sector(path, scale):
    '''Write the synthetic observable sector at `scale` to the directory `path`, load it and return it.
    Its predictions are linear in the parameters.'''
    rng = np.random.default_rng(20261103)
    keys = [('', '', 'RR')] + [
        ('', name, 'R' + part)
        for s in SECTORS for name, part in get_wc_basis(EFT, BASIS, s) if name in PARAMETERS
    ]
    observable_central = {
        str(key): SM_VALUES if key == ('', '', 'RR') else list(rng.normal(size=len(SM_VALUES)))
        for key in keys
    }
    json_data = {
        '$schema': 'https://json.schemastore.org/popxf-1.0.json',
        'metadata': {
            'observable_names': list(MEASUREMENTS),
            'parameters': PARAMETERS,
            'basis': {'wcxf': {'eft': EFT, 'basis': BASIS, 'sectors': SECTORS}},
            'scale': scale,
        },
        'data': {'observable_central': observable_central},
    }
    file = path / f'{NAME}.json'
    file.write_text(json.dumps(json_data))
    ObservableSector.load(str(file))
    return ObservableSector.get(NAME)


@pytest.fixture
def sector(tmp_path, jelli_registries):
    '''The synthetic observable sector at 4.8 GeV, loaded into empty registries.'''
    return _load_sector(tmp_path, SCALE)


def _reference(sector, eft, basis):
    '''Return the input Wilson coefficients and the evolution matrices of `sector`, built sector by
    sector: for each sector, the rows of the parameters, taken by name from the matrix of
    `run_and_match(..., sector_out=s)`, go into the columns of its input sector. All other entries
    are zero.'''
    wcs_in, offset = [], {}
    for s in SECTORS_IN[eft]:
        offset[s] = len(wcs_in)
        wcs_in += get_wc_basis(eft, basis, s)
    rows = sector.keys_pars[1:]
    scales = get_scales(eft, basis)
    matrices = np.zeros((len(scales), len(rows), len(wcs_in)))
    for k, scale_in in enumerate(scales):
        for s in SECTORS:
            block = run_and_match(
                eft_in=eft, eft_out=EFT, basis_in=basis, basis_out=BASIS,
                scale_in=scale_in, scale_out=sector.scale, sector_out=s,
            )
            basis_out = get_wc_basis(EFT, BASIS, s)
            start = offset[SECTOR_IN[eft][s]]
            for i, wc in enumerate(rows):
                if wc in basis_out:
                    matrices[k, i, start:start + block.shape[1]] = block[basis_out.index(wc)]
    return wcs_in, matrices


def _predictions(sector, evolved):
    '''The predictions for the evolved parameters `evolved`, ordered as `keys_pars[1:]`, from the
    linear polynomials of the sector.'''
    rows = sector.keys_pars[1:]
    monomials = []
    for w1, w2, c in sector.keys_coeff_observable:
        assert w1 == ''  # the polynomials are linear
        monomials.append(1.0 if w2 == '' else evolved[rows.index((w2, c[1]))])
    return np.array(monomials) @ np.asarray(sector.observable_central)


def test_evolution_pairs(sector):
    '''The sector has evolution matrices for the four (eft, basis) pairs below and above the
    electroweak scale, and for any other installed rgevolve basis of these EFTs. The rows of the
    matrices are the parameters of three sectors.'''
    pairs = [(eft, basis) for eft, matrices in sector.evolution_matrices.items() for basis in matrices]
    assert set(EVOLUTIONS) <= set(pairs)
    assert sector.basis_mode == 'rgevolve'
    assert {name for name, _ in sector.keys_pars[1:]} == set(PARAMETERS)


@pytest.mark.parametrize('eft, basis', EVOLUTIONS)
def test_evolution_matrices(sector, eft, basis):
    '''The sector indices point at the input coefficients in the full basis, and the evolution
    matrices equal the reference built sector by sector.'''
    wcs_in, reference = _reference(sector, eft, basis)
    full_basis = get_wc_basis(eft, basis)
    assert [full_basis[i] for i in sector.sector_indices[eft][basis]] == wcs_in
    np.testing.assert_array_equal(sector.evolution_matrices[eft][basis], reference)


def test_evolution_matrices_at_reference_scale(tmp_path, jelli_registries):
    '''The sector at the reference scale of the WET: from the WET JMS basis at that scale, the
    evolution matrix is the change of basis alone. All evolution matrices equal the reference.
    rgevolve-core 0.2.0 failed here with a TypeError; 0.2.1 fixed it.'''
    sector = _load_sector(tmp_path, float(reference_scale['WET']))
    for eft, basis in EVOLUTIONS:
        _, reference = _reference(sector, eft, basis)
        np.testing.assert_array_equal(sector.evolution_matrices[eft][basis], reference)


def test_smeft_structure(sector):
    '''In the SMEFT matrices, the parameters of sbemu and sbmue depend on the columns of the SMEFT
    sector mue, and those of etauemu only on the columns of the SMEFT sector etauemu.'''
    matrices = sector.evolution_matrices['SMEFT']['Warsaw']
    n_etauemu = len(get_wc_basis('SMEFT', 'Warsaw', 'etauemu'))  # the first input sector
    for s in SECTORS:
        rows = [i for i, wc in enumerate(sector.keys_pars[1:]) if wc in get_wc_basis(EFT, BASIS, s)]
        mue_columns = np.abs(matrices[:, rows, n_etauemu:]).max()
        etauemu_columns = np.abs(matrices[:, rows, :n_etauemu]).max()
        if s == 'etauemu':
            assert mue_columns == 0 and etauemu_columns > 0
        else:
            assert mue_columns > 0 and etauemu_columns == 0


def test_smeft_likelihood(sector):
    '''In a likelihood in the SMEFT Warsaw basis, the jitted prediction equals the polynomial of
    the evolved coefficients, and the jitted neg-logL and its gradient agree with the Gaussian
    log-likelihood of these predictions.'''
    Measurement('wet_sector measurements', [
        {'distribution_type': 'NormalDistribution', 'observables': [name],
         'central_value': central, 'standard_deviation': std}
        for name, (central, std) in MEASUREMENTS.items()
    ])
    ExperimentalCorrelations.load()
    gl = GlobalLikelihood(eft='SMEFT', basis='Warsaw')

    wcs_in, reference = _reference(sector, 'SMEFT', 'Warsaw')
    matrix = reference[list(get_scales('SMEFT', 'Warsaw')).index(SMEFT_SCALE)]
    full_basis = get_wc_basis('SMEFT', 'Warsaw')
    positions = [full_basis.index(wc) for wc in wcs_in]

    # a random point in the input coefficients, scaled so that the evolved parameters are of order one
    x = np.random.default_rng(1).normal(size=len(wcs_in))
    x /= np.abs(matrix @ x).max()
    par_array = np.zeros(len(full_basis))
    par_array[positions] = x
    expected = _predictions(sector, matrix @ x)
    assert np.abs(expected - SM_VALUES).max() > 0.1  # the point moves the predictions
    predicted, _ = jax.jit(sector.prediction)(
        jnp.array(par_array), SMEFT_SCALE, sector.get_prediction_data('SMEFT', 'Warsaw')
    )
    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)

    # the neg-logL of three parameters with large columns, compared between two points
    par_list = [wcs_in[j] for j in np.argsort(-np.abs(matrix).max(axis=0))[:3]]
    columns = [wcs_in.index(p) for p in par_list]
    neg_log_likelihood, data = gl.get_negative_log_likelihood(par_list, 'global')
    values = [x[columns], -0.5 * x[columns]]
    central, std = np.array(list(MEASUREMENTS.values())).T
    chi2_half = [
        np.sum(((_predictions(sector, matrix[:, columns] @ v) - central) / std) ** 2) / 2
        for v in values
    ]
    nll = [float(jax.jit(neg_log_likelihood)(jnp.array(v), SMEFT_SCALE, data)) for v in values]
    np.testing.assert_allclose(nll[1] - nll[0], chi2_half[1] - chi2_half[0], rtol=1e-5)
    gradient = jax.jit(jax.grad(neg_log_likelihood))(jnp.array(values[0]), SMEFT_SCALE, data)
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0)


def test_deprecated_helpers(sector):
    '''get_wc_mask and get_sector_indices still work, and warn that they are deprecated.'''
    # imported here, so that removing the helpers only removes this test
    from jelli.utils.par_helpers import get_sector_indices, get_wc_mask
    with pytest.warns(DeprecationWarning, match='get_sector_indices'):
        indices = get_sector_indices(EFT, BASIS, sector.sectors)
    np.testing.assert_array_equal(indices, sector.sector_indices[EFT][BASIS])
    with pytest.warns(DeprecationWarning, match='get_wc_mask'):
        mask = get_wc_mask(EFT, BASIS, 'sbemu', [('C9_bsmue', 'R')])
    assert [wc for wc, selected in zip(get_wc_basis(EFT, BASIS, 'sbemu'), mask) if selected] == [('C9_bsmue', 'R')]
