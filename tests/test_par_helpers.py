import unittest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jelli.core.observable_sector import interpolate_rg_evolution
from jelli.utils.par_helpers import get_par_monomial_indices, get_active_monomial_indices

class TestParHelpers(unittest.TestCase):

    def test_get_active_monomial_indices(self):
        # a sector with the parameters b, c, d of a full parameter array (a, b, c, d, e);
        # the evolution mixes b into c at the lower scale and is the identity at the upper one
        keys_pars = [('', 'R'), ('b', 'R'), ('c', 'R'), ('d', 'R')]
        keys_coeff = sorted(
            (keys_pars[k][0], keys_pars[l][0], keys_pars[k][1] + keys_pars[l][1])
            for k in range(4) for l in range(k, 4)
        )
        sector_indices = np.array([1, 2, 3])
        evolution_matrices = np.array([
            [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
            np.eye(3),
        ], dtype=np.float32)
        evolution_scales = jnp.array([10.0, 100.0, np.nan], dtype=jnp.float32)  # as in the prediction data
        pmi = get_par_monomial_indices(keys_pars, keys_coeff)

        def monomials(par_array, scale):
            evolved = interpolate_rg_evolution(
                jnp.array(par_array[sector_indices]), scale, jnp.array(evolution_matrices), evolution_scales
            )
            pars = np.concatenate([[1.0], np.array(evolved)])
            return np.outer(pars, pars).ravel()[pmi]

        # a and b can be nonzero; a is not a parameter of the sector, and c is reached through the mixing
        par_indices = np.array([0, 1])
        active = get_active_monomial_indices(keys_pars, keys_coeff, sector_indices, evolution_matrices, par_indices)
        expected = [k for k, (w1, w2, _) in enumerate(keys_coeff) if {w1, w2} <= {'', 'b', 'c'}]
        np.testing.assert_array_equal(active, expected)

        # every dropped monomial vanishes exactly, at the tabulated scales and between them
        dropped = np.setdiff1d(np.arange(len(keys_coeff)), active)
        rng = np.random.default_rng(0)
        for scale in [10.0, 31.6, 100.0]:
            for _ in range(5):
                par_array = np.zeros(5)
                par_array[par_indices] = rng.standard_normal(len(par_indices))
                self.assertTrue(np.all(monomials(par_array, scale)[dropped] == 0))

        # a rule by parameter name keeps only the monomials of b and the constant, and so would
        # drop monomials that are nonzero below the upper scale
        by_name = [k for k, (w1, w2, _) in enumerate(keys_coeff) if {w1, w2} <= {'', 'b'}]
        missed = np.setdiff1d(active, by_name)
        self.assertTrue(np.all(monomials(par_array, 10.0)[missed] != 0))

        # no parameter of the sector can be nonzero: only the constant monomial is left
        np.testing.assert_array_equal(
            get_active_monomial_indices(keys_pars, keys_coeff, sector_indices, evolution_matrices, np.array([0, 4])),
            [keys_coeff.index(('', '', 'RR'))],
        )
        # all parameters can be nonzero: every monomial is kept
        np.testing.assert_array_equal(
            get_active_monomial_indices(keys_pars, keys_coeff, sector_indices, evolution_matrices, np.arange(5)),
            np.arange(len(keys_coeff)),
        )
