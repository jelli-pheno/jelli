import unittest
import numpy as np
import scipy as sp
from jelli.utils.distributions import (
    combine_normal_distributions,
    get_distribution_support,
    combine_distributions_numerically,
    get_ppf_numerical_distribution,
    get_mode_and_uncertainty,
    get_distribution_samples,
)

class TestDistributions(unittest.TestCase):

    def test_combine_normal_distributions(self):
        measurement_name = np.array(['measurement1', 'measurement2'])
        observables = np.array(['observable1', 'observable1'])
        observable_indices = np.array([3, 3])
        central_value = np.array([1.0, 2.0])
        standard_deviation = np.array([0.1, 0.2])
        combined = combine_normal_distributions(
            measurement_name=measurement_name,
            observables=observables,
            observable_indices=observable_indices,
            central_value=central_value,
            standard_deviation=standard_deviation
        )

        for key, value in combined.items():
            self.assertEqual(len(value), 1, f"Expected {key} to have length 1, got {len(value)}.")

        self.assertEqual(combined['measurement_name'][0], 'measurement1, measurement2')
        self.assertEqual(combined['observables'][0], 'observable1')
        self.assertEqual(combined['observable_indices'][0], 3)
        self.assertAlmostEqual(combined['central_value'][0], 1.2)
        self.assertAlmostEqual(combined['standard_deviation'][0], 0.08944271909999159)


    def test_get_distribution_support(self):
        # NumericalDistribution
        xp = np.array([
            [0., 0.5, 1.0],
            [-0.5, 0.0, 0.5],
            [0.1, 0.2, 0.3]
        ])
        dist_type = 'NumericalDistribution'
        dist_info = {'x': xp}
        support_min, support_max = get_distribution_support(dist_type, dist_info)
        np.testing.assert_almost_equal(support_min, np.array([0.0, -0.5, 0.1]))
        np.testing.assert_almost_equal(support_max, np.array([1.0, 0.5, 0.3]))

        # NormalDistribution
        central_value = np.array([1.0, 2.0, 3.0])
        standard_deviation = np.array([0.1, 0.2, 0.3])
        dist_type = 'NormalDistribution'
        dist_info = {
            'central_value': central_value,
            'standard_deviation': standard_deviation
        }
        support_min, support_max = get_distribution_support(dist_type, dist_info)
        np.testing.assert_almost_equal(support_min, np.array([0.4, 0.8, 1.2]))
        np.testing.assert_almost_equal(support_max, np.array([1.6, 3.2, 4.8]))

        # HalfNormalDistribution
        standard_deviation = np.array([0.1, 0.2, 0.3])
        dist_type = 'HalfNormalDistribution'
        dist_info = {'standard_deviation': standard_deviation}
        support_min, support_max = get_distribution_support(dist_type, dist_info)
        np.testing.assert_equal(support_min, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(support_max, np.array([0.6, 1.2, 1.8]))

        # GammaDistributionPositive
        a = np.array([1.0, 2.0, 3.0])
        loc = np.array([0.0, -0.2, -0.4])
        scale = np.array([0.2, 0.5, 0.7])
        dist_type = 'GammaDistributionPositive'
        dist_info = {'a': a, 'loc': loc, 'scale': scale}
        support_min, support_max = get_distribution_support(dist_type, dist_info)
        np.testing.assert_almost_equal(support_min, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(support_max, np.array([4.1446532, 11.8029528, 18.2861401]))

    def test_combine_distributions_numerically(self):
        # compare combining normal distributions numerically to analytical result
        constraints = {
            'NormalDistribution': {
                'measurement_name': np.array(['measurement1', 'measurement2']),
                'observables': np.array(['observable1', 'observable1']),
                'observable_indices': np.array([3, 3]),
                'central_value': np.array([1.0, 2.0]),
                'standard_deviation': np.array([0.1, 0.2])
            }
        }
        numerical_distribution = combine_distributions_numerically(constraints)
        xp = numerical_distribution['x']
        y = numerical_distribution['y']
        dx = xp[1] - xp[0]

        # check the output distribution is normalized
        self.assertAlmostEqual(np.sum(y*dx), 1.0, places=6)

        # check the central value and standard deviation
        central_value = np.sum(xp * y) * dx
        standard_deviation = np.sqrt(np.sum((xp - central_value) ** 2 * y) * dx)
        self.assertAlmostEqual(central_value, 1.2, places=6)
        self.assertAlmostEqual(standard_deviation, 0.08944271909999159, places=6)

        # check the rest of the distribution information
        self.assertEqual(numerical_distribution['measurement_name'][0], 'measurement1, measurement2')
        self.assertEqual(numerical_distribution['observables'][0], 'observable1')
        self.assertEqual(numerical_distribution['observable_indices'][0], 3)

    def test_get_ppf_numerical_distribution(self):
        central_value = 5.0
        standard_deviation = 1.0
        xp = np.linspace(0, 10, 10000)
        fp = sp.stats.norm.pdf(xp, loc=central_value, scale=standard_deviation)
        ppf_numerical_distribution = get_ppf_numerical_distribution(xp, fp)
        p_test = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        expected_values = sp.stats.norm.ppf(p_test, loc=central_value, scale=standard_deviation)
        computed_values = ppf_numerical_distribution(p_test)
        np.testing.assert_almost_equal(computed_values, expected_values, decimal=5)

        a = 3.0
        loc = 0.0
        scale = 1.5
        xp = np.linspace(0, 40, 10000)
        fp = sp.stats.gamma.pdf(xp, a=a, loc=loc, scale=scale)
        ppf_numerical_distribution = get_ppf_numerical_distribution(xp, fp)
        p_test = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        expected_values = sp.stats.gamma.ppf(p_test, a=a, loc=loc, scale=scale)
        computed_values = ppf_numerical_distribution(p_test)
        np.testing.assert_almost_equal(computed_values, expected_values, decimal=5)

    def test_get_mode_and_uncertainty(self):
        # NormalDistribution
        dist_type = 'NormalDistribution'
        dist_info = {
            'central_value': np.array([1.0, 2.0]),
            'standard_deviation': np.array([0.1, 0.2])
        }
        mode, uncertainty = get_mode_and_uncertainty(dist_type, dist_info)
        self.assertEqual(len(mode), 2)
        self.assertEqual(len(uncertainty), 2)
        np.testing.assert_almost_equal(mode, dist_info['central_value'])
        np.testing.assert_almost_equal(uncertainty, dist_info['standard_deviation'])

        # HalfNormalDistribution
        dist_type = 'HalfNormalDistribution'
        dist_info = {
            'standard_deviation': np.array([0.2, 2.0])
        }
        mode, uncertainty = get_mode_and_uncertainty(dist_type, dist_info)
        self.assertEqual(len(mode), 2)
        self.assertEqual(len(uncertainty), 2)
        np.testing.assert_equal(mode, np.array([np.nan, np.nan]))
        np.testing.assert_almost_equal(uncertainty, dist_info['standard_deviation'] * 1.96)

        # GammaDistributionPositive
        dist_type = 'GammaDistributionPositive'
        dist_info = {
            'a': np.array([3, 100, 0.5]),
            'loc': np.array([-1, 2.5, 1]),
            'scale': np.array([1, 0.5, 2]),
        }
        mode, uncertainty = get_mode_and_uncertainty(dist_type, dist_info)
        self.assertEqual(len(mode), 3)
        self.assertEqual(len(uncertainty), 3)
        np.testing.assert_almost_equal(mode, np.array([np.nan, 52., np.nan]))
        np.testing.assert_almost_equal(uncertainty, np.array([5.41000494, 4.97493719, 4.84145882]))

        # NumericalDistribution
        central_value = np.array([[0.0], [6.4]])
        standard_deviation = np.array([[1.0], [1.2]])  # upper limit and Gaussian expected
        xp = np.broadcast_to(np.linspace(0, 10, 10000), (2, 10000))
        fp = sp.stats.norm.pdf(xp, loc=central_value, scale=standard_deviation)
        dist_type = 'NumericalDistribution'
        dist_info = {
            'x': xp,
            'y': fp,
            'log_y': np.log(fp)
        }
        mode, uncertainty = get_mode_and_uncertainty(dist_type, dist_info)
        self.assertEqual(len(mode), 2)
        self.assertEqual(len(uncertainty), 2)
        np.testing.assert_almost_equal(mode, np.array([np.nan, 6.4]))
        np.testing.assert_almost_equal(uncertainty, np.array([sp.stats.halfnorm(0.0, 1.0).ppf(0.95), 1.2]), decimal=5)

    def test_get_distribution_samples(self):
        seed = 42
        n_samples = 1000000

        # GammaDistributionPositive
        a = np.array([1.0, 2.0, 20.0])
        loc = np.array([3.0, 2.0, 1.0])
        scale = np.array([1.0, 2.0, 0.5])
        dist_info = {
            'a': a,
            'loc': loc,
            'scale': scale
        }
        dist_type = 'GammaDistributionPositive'
        samples = get_distribution_samples(dist_type, dist_info, n_samples, seed=seed)
        self.assertEqual(samples.shape, (len(a), n_samples))
        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertTrue(np.all(samples >= 0))
        for i, sample in enumerate(samples):
            mean_expected = a[i] * scale[i] + loc[i]
            var_expected = a[i] * (scale[i] ** 2)
            mean_computed = np.mean(sample)
            var_computed = np.std(sample)**2
            np.testing.assert_almost_equal(mean_computed, mean_expected, decimal=2)
            np.testing.assert_almost_equal(var_computed, var_expected, decimal=2)

        # NumericalDistribution
        central_value_expected = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).reshape(-1, 1)
        standard_deviation_expected = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        xp = np.broadcast_to(np.linspace(-10, 10, 10000), (len(central_value_expected), 10000))
        fp = sp.stats.norm.pdf(xp, loc=central_value_expected, scale=standard_deviation_expected)
        dist_type = 'NumericalDistribution'
        dist_info = {
            'x': xp,
            'y': fp,
        }
        samples = get_distribution_samples(
            dist_type,
            dist_info,
            n_samples,
            seed=seed
        )
        self.assertEqual(samples.shape, (len(central_value_expected), n_samples))
        self.assertTrue(np.all(np.isfinite(samples)))
        central_value_computed = np.mean(samples, axis=1)
        standard_deviation_computed = np.std(samples, axis=1)
        np.testing.assert_almost_equal(central_value_expected.reshape(-1), central_value_computed, decimal=2)
        np.testing.assert_almost_equal(standard_deviation_expected.reshape(-1), standard_deviation_computed, decimal=2)

        # NormalDistribution
        central_value_expected = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        standard_deviation_expected = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.8])
        dist_type = 'NormalDistribution'
        dist_info = {
            'central_value': central_value_expected,
            'standard_deviation': standard_deviation_expected
        }
        samples = get_distribution_samples(
            dist_type,
            dist_info,
            n_samples,
            seed=seed
        )
        self.assertEqual(samples.shape, (len(central_value_expected), n_samples))
        self.assertTrue(np.all(np.isfinite(samples)))
        central_value_computed = np.mean(samples, axis=1)
        standard_deviation_computed = np.std(samples, axis=1)
        np.testing.assert_almost_equal(central_value_expected, central_value_computed, decimal=2)
        np.testing.assert_almost_equal(standard_deviation_expected, standard_deviation_computed, decimal=2)

        # MultivariateNormalDistribution
        central_value_expected = np.array([
            [0.0, 1.0],
            [2.0, -1.0],
            [-1.0, 3.0]
        ])
        standard_deviation_expected = np.array([
            [1.0, 0.5],
            [0.3, 0.8],
            [0.6, 0.4]
        ])
        n_constraints = len(central_value_expected)
        dim = len(central_value_expected[0])
        correlation_matrices = []
        inverse_correlation = []
        for _ in range(n_constraints):
            A = np.random.rand(dim, dim)
            corr = np.dot(A, A.T)
            diag = np.sqrt(np.diag(corr))
            corr = corr / np.outer(diag, diag)
            correlation_matrices.append(corr)
            inverse_correlation.append(np.linalg.inv(corr))
        dist_type = 'MultivariateNormalDistribution'
        dist_info = {
            'central_value': central_value_expected,
            'standard_deviation': standard_deviation_expected,
            'inverse_correlation': inverse_correlation
        }
        samples = get_distribution_samples(dist_type, dist_info, n_samples, seed=seed)
        self.assertEqual(len(samples), n_constraints)
        for i in range(n_constraints):
            sample = samples[i]
            self.assertEqual(sample.shape, (dim, n_samples))
            self.assertTrue(np.all(np.isfinite(sample)))
            central_value_computed = np.mean(sample, axis=1)
            np.testing.assert_almost_equal(central_value_computed, central_value_expected[i], decimal=2)
            cov_computed = np.cov(sample)
            corr = np.linalg.inv(inverse_correlation[i])
            cov_expected = corr * np.outer(standard_deviation_expected[i], standard_deviation_expected[i])
            np.testing.assert_almost_equal(cov_computed, cov_expected, decimal=2)

        # HalfNormalDistribution
        standard_deviation = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.8])
        dist_type = 'HalfNormalDistribution'
        dist_info = {
            'standard_deviation': standard_deviation
        }
        samples = get_distribution_samples(
            dist_type,
            dist_info,
            n_samples,
            seed=seed
        )
        self.assertEqual(samples.shape, (len(standard_deviation), n_samples))
        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertTrue(np.all(samples >= 0))
        central_value_computed = np.mean(samples, axis=1)
        standard_deviation_computed = np.std(samples, axis=1)
        central_value_expected = standard_deviation * np.sqrt(2 / np.pi)
        standard_deviation_expected = standard_deviation * np.sqrt(1 - 2 / np.pi)
        np.testing.assert_almost_equal(central_value_expected, central_value_computed, decimal=2)
        np.testing.assert_almost_equal(standard_deviation_expected, standard_deviation_computed, decimal=2)
