from typing import List, Dict, Tuple, Callable, Union, Optional
import numpy as np
import scipy as sp
from jax import vmap, numpy as jnp, scipy as jsp
from functools import partial
from .probability import GammaDistribution, NormalDistribution, NumericalDistribution, _convolve_numerical

LOG_ZERO = -100.0 # exp(-100) = 3.7e-44 is a good approximation of zero in a PDF

def convert_GeneralGammaDistributionPositive(a, loc, scale, gaussian_standard_deviation):
    loc_scaled = loc/scale
    if gaussian_standard_deviation == 0:
        distribution_type = 'GammaDistributionPositive'
        parameters = {'a': a, 'loc': loc_scaled, 'scale': 1}
    else:
        distribution_type = 'NumericalDistribution'
        gamma_unscaled = GammaDistribution(a = a, loc = loc_scaled, scale = 1)
        norm_bg = NormalDistribution(0, gaussian_standard_deviation)
        numerical = [NumericalDistribution.from_pd(p, nsteps=1000) for p in [gamma_unscaled, norm_bg]]
        num_unscaled = _convolve_numerical(numerical, central_values='sum')
        x = np.array(num_unscaled.x)
        y = np.array(num_unscaled.y_norm)
        if loc_scaled in x:
            to_mirror = y[x<=loc_scaled][::-1]
            y_pos = y[len(to_mirror)-1:len(to_mirror)*2-1]
            y[len(to_mirror)-1:len(to_mirror)*2-1] += to_mirror[:len(y_pos)]
        else:
            to_mirror = y[x<loc_scaled][::-1]
            y_pos = y[len(to_mirror):len(to_mirror)*2]
            y[len(to_mirror):len(to_mirror)*2] += to_mirror[:len(y_pos)]
        y = y[x >= 0]
        x = x[x >= 0]
        if x[0] != 0:  #  make sure the PDF at 0 exists
            x = np.insert(x, 0, 0.)  # add 0 as first element
            y = np.insert(y, 0, y[0])  # copy first element
        x = x * scale
        y = np.maximum(0, y)  # make sure PDF is positive
        y = y /  np.trapz(y, x=x)  # normalize PDF to 1
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            log_y = np.log(y)
        # replace -np.inf with a large negative number
        log_y[np.isneginf(log_y)] = LOG_ZERO
        parameters = {
            'x': x,
            'y': y,
            'log_y': log_y,
        }
    return distribution_type, parameters

interp_log_pdf = partial(jnp.interp, left=LOG_ZERO, right=LOG_ZERO)

def logpdf_numerical_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        x: jnp.array,
        log_y: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logpdf_numerical_distribution(predictions, observable_indices, x, log_y)

def logpdf_numerical_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        x: jnp.array,
        log_y: jnp.array,
    ) -> jnp.array:
    logpdf_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    logpdf = vmap(interp_log_pdf)(predictions, x, log_y)
    logpdf_total = logpdf_total.at[observable_indices].add(logpdf)
    return logpdf_total

def logpdf_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logpdf_normal_distribution(predictions, observable_indices, mean, std)

def logpdf_normal_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    logpdf_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    logpdf = jsp.stats.norm.logpdf(predictions, loc=mean, scale=std)
    logpdf_total = logpdf_total.at[observable_indices].add(logpdf)
    return logpdf_total

def logpdf_folded_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logpdf_folded_normal_distribution(predictions, observable_indices, mean, std)

def logpdf_folded_normal_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    logpdf_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    folded_logpdf = jnp.log(
        jsp.stats.norm.pdf(predictions, loc=mean, scale=std)
        + jsp.stats.norm.pdf(predictions, loc=-mean, scale=std)
    )
    logpdf = jnp.where(predictions >= 0, folded_logpdf, LOG_ZERO)
    logpdf_total = logpdf_total.at[observable_indices].add(logpdf)
    return logpdf_total

def logpdf_half_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return logpdf_folded_normal_distribution_summed(predictions, selector_matrix, observable_indices, 0, std)

def logpdf_half_normal_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return logpdf_folded_normal_distribution(predictions, observable_indices, 0, std)

def logpdf_gamma_distribution_positive_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        a: jnp.array,
        loc: jnp.array,
        scale: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logpdf_gamma_distribution_positive(predictions, observable_indices, a, loc, scale)

def logpdf_gamma_distribution_positive(
        predictions: jnp.array,
        observable_indices: jnp.array,
        a: jnp.array,
        loc: jnp.array,
        scale: jnp.array,
    ) -> jnp.array:
    logpdf_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    log_pdf_scale = jnp.log(1/(1-jsp.stats.gamma.cdf(0, a, loc=loc, scale=scale)))
    positive_logpdf = jsp.stats.gamma.logpdf(
        predictions, a, loc=loc, scale=scale
    ) + log_pdf_scale
    logpdf = jnp.where(predictions>=0, positive_logpdf, LOG_ZERO)
    logpdf_total = logpdf_total.at[observable_indices].add(logpdf)
    return logpdf_total

def logpdf_multivariate_normal_distribution_summed(
    predictions: jnp.array,
    selector_matrix: jnp.array,
    observable_indices: List[jnp.array],
    mean: List[jnp.array],
    standard_deviation: List[jnp.array],
    inverse_correlation: List[jnp.array],
    logpdf_normalization_per_observable: List[jnp.array],
) -> jnp.array:
    logpdf_rows = []
    for i in range(len(observable_indices)):
        d = (jnp.take(predictions, observable_indices[i]) - mean[i]) / standard_deviation[i]
        n_obs = d.shape[0]
        logpdf = -0.5 * jnp.dot(d, jnp.dot(inverse_correlation[i], d)) + n_obs * logpdf_normalization_per_observable[i]
        logpdf_rows.append(logpdf)
    logpdf_total = jnp.stack(logpdf_rows)
    return selector_matrix @ logpdf_total

def logpdf_multivariate_normal_distribution(
    predictions: jnp.array,
    observable_indices: List[jnp.array],
    mean: List[jnp.array],
    standard_deviation: List[jnp.array],
    inverse_correlation: List[jnp.array],
    logpdf_normalization_per_observable: List[jnp.array],
) -> List[jnp.array]:
    logpdfs = []
    for i in range(len(observable_indices)):
        logpdf_total = jnp.zeros_like(predictions)
        d = (jnp.take(predictions, observable_indices[i]) - mean[i]) / standard_deviation[i]
        logpdf = -0.5 * d * jnp.dot(inverse_correlation[i], d) + logpdf_normalization_per_observable[i]
        logpdf_total = logpdf_total.at[observable_indices[i]].add(logpdf)
        logpdfs.append(logpdf_total)
    return jnp.stack(logpdfs)

def logL_numerical_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        x: jnp.array,
        log_y: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logL_numerical_distribution(predictions, observable_indices, x, log_y)

def logL_numerical_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        x: jnp.array,
        log_y: jnp.array,
    ) -> jnp.array:
    logL_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    logL = vmap(interp_log_pdf)(predictions, x, log_y - jnp.max(log_y, axis=1, keepdims=True))
    logL_total = logL_total.at[observable_indices].add(logL)
    return logL_total

def logL_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logL_normal_distribution(predictions, observable_indices, mean, std)

def logL_normal_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        mean: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    logL_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    logL = -0.5 * ((predictions-mean)/std)**2
    logL_total = logL_total.at[observable_indices].add(logL)
    return logL_total

def logL_half_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logL_half_normal_distribution(predictions, observable_indices, std)

def logL_half_normal_distribution(
        predictions: jnp.array,
        observable_indices: jnp.array,
        std: jnp.array,
    ) -> jnp.array:
    logL_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    logL = -0.5 * (predictions/std)**2
    logL = jnp.where(predictions>=0, logL, LOG_ZERO)
    logL_total = logL_total.at[observable_indices].add(logL)
    return logL_total

def logL_gamma_distribution_positive_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: jnp.array,
        a: jnp.array,
        loc: jnp.array,
        scale: jnp.array,
    ) -> jnp.array:
    return selector_matrix @ logL_gamma_distribution_positive(predictions, observable_indices, a, loc, scale)

def logL_gamma_distribution_positive(
        predictions: jnp.array,
        observable_indices: jnp.array,
        a: jnp.array,
        loc: jnp.array,
        scale: jnp.array,
    ) -> jnp.array:
    logL_total = jnp.zeros_like(predictions)
    predictions = jnp.take(predictions, observable_indices)
    mode = jnp.maximum(loc + (a-1)*scale, 0)
    logL_pred = (a-1)*jnp.log((predictions-loc)/scale) - (predictions-loc)/scale
    logL_mode = (a-1)*jnp.log((mode-loc)/scale) - (mode-loc)/scale
    logL = jnp.where(predictions>=0, logL_pred-logL_mode, LOG_ZERO)
    logL_total = logL_total.at[observable_indices].add(logL)
    return logL_total

def logL_multivariate_normal_distribution_summed(
        predictions: jnp.array,
        selector_matrix: jnp.array,
        observable_indices: List[jnp.array],
        mean: List[jnp.array],
        standard_deviation: List[jnp.array],
        inverse_correlation: List[jnp.array],
    ) -> jnp.array:
    logL_rows = []
    for i in range(len(observable_indices)):
        d = (jnp.take(predictions, observable_indices[i]) - mean[i]) / standard_deviation[i]
        logL = -0.5 * jnp.dot(d, jnp.dot(inverse_correlation[i], d))
        logL_rows.append(logL)
    logL_total = jnp.stack(logL_rows)
    return selector_matrix @ logL_total

def logL_multivariate_normal_distribution(
        predictions: jnp.array,
        observable_indices: List[jnp.array],
        mean: List[jnp.array],
        standard_deviation: List[jnp.array],
        inverse_correlation: List[jnp.array],
    ) -> jnp.array:
    logLs = []
    for i in range(len(observable_indices)):
        logL_total = jnp.zeros_like(predictions)
        d = (jnp.take(predictions, observable_indices[i]) - mean[i]) / standard_deviation[i]
        logL = -0.5 * d * jnp.dot(inverse_correlation[i], d)
        logL_total = logL_total.at[observable_indices[i]].add(logL)
        logLs.append(logL_total)
    return jnp.stack(logLs)

logpdf_functions_summed = {
    'NumericalDistribution': logpdf_numerical_distribution_summed,
    'NormalDistribution': logpdf_normal_distribution_summed,
    'HalfNormalDistribution': logpdf_half_normal_distribution_summed,
    'GammaDistributionPositive': logpdf_gamma_distribution_positive_summed,
    'MultivariateNormalDistribution': logpdf_multivariate_normal_distribution_summed,
}

logpdf_functions = {
    'NumericalDistribution': logpdf_numerical_distribution,
    'NormalDistribution': logpdf_normal_distribution,
    'HalfNormalDistribution': logpdf_half_normal_distribution,
    'GammaDistributionPositive': logpdf_gamma_distribution_positive,
    'MultivariateNormalDistribution': logpdf_multivariate_normal_distribution,
}

logL_functions = {
    'NumericalDistribution': logL_numerical_distribution,
    'NormalDistribution': logL_normal_distribution,
    'HalfNormalDistribution': logL_half_normal_distribution,
    'GammaDistributionPositive': logL_gamma_distribution_positive,
    'MultivariateNormalDistribution': logL_multivariate_normal_distribution,
}

logL_functions_summed = {
    'NumericalDistribution': logL_numerical_distribution_summed,
    'NormalDistribution': logL_normal_distribution_summed,
    'HalfNormalDistribution': logL_half_normal_distribution_summed,
    'GammaDistributionPositive': logL_gamma_distribution_positive_summed,
    'MultivariateNormalDistribution': logL_multivariate_normal_distribution_summed,
}

def cov_coeff_to_cov_obs(par_monomials, cov_th_scaled): # TODO (maybe) optimize
    n_sectors = len(par_monomials)

    cov = np.empty((n_sectors,n_sectors), dtype=object).tolist()

    for i in range(n_sectors):
        for j in range(n_sectors):
            if i>= j:
                cov[i][j] = jnp.einsum('ijkl,k,l->ij',cov_th_scaled[i][j],par_monomials[i],par_monomials[j])
            else:
                shape = cov_th_scaled[j][i].shape
                cov[i][j] = jnp.zeros((shape[1], shape[0]))
    cov_matrix_tril = jnp.tril(jnp.block(cov))
    return cov_matrix_tril + cov_matrix_tril.T - jnp.diag(jnp.diag(cov_matrix_tril))

def logpdf_correlated_sectors_summed(
    predictions_scaled: jnp.array,
    std_sm_exp: jnp.array,
    selector_matrix: jnp.array,
    observable_indices: List[jnp.array],
    exp_central_scaled: jnp.array,
    cov_matrix_exp_scaled: jnp.array,
    cov_matrix_th_scaled: jnp.array,
) -> jnp.array:

    cov_scaled = cov_matrix_th_scaled + cov_matrix_exp_scaled
    std_scaled = jnp.sqrt(jnp.diag(cov_scaled))
    std = std_scaled  * std_sm_exp
    C = cov_scaled / jnp.outer(std_scaled, std_scaled)
    D = (predictions_scaled - exp_central_scaled)/std_scaled

    logpdf_rows = []
    for i in range(len(observable_indices)):

        d = jnp.take(D, observable_indices[i])
        c = jnp.take(jnp.take(C, observable_indices[i], axis=0), observable_indices[i], axis=1)

        logdet_corr = jnp.linalg.slogdet(c)[1]
        logprod_std2 = 2 * jnp.sum(jnp.log(jnp.take(std, observable_indices[i])))

        logpdf = -0.5 * (
            jnp.dot(d, jsp.linalg.cho_solve(jsp.linalg.cho_factor(c), d))
            + logdet_corr
            + logprod_std2
            + len(d) * jnp.log(2 * jnp.pi)
        )
        logpdf = jnp.where(jnp.isnan(logpdf), len(d)*LOG_ZERO, logpdf)
        logpdf_rows.append(logpdf)
    logpdf_total = jnp.array(logpdf_rows)
    return selector_matrix @ logpdf_total

def logpdf_correlated_sectors(
    predictions_scaled: jnp.array,
    std_sm_exp: jnp.array,
    observable_indices: List[jnp.array],
    exp_central_scaled: jnp.array,
    cov_matrix_exp_scaled: jnp.array,
    cov_matrix_th_scaled: jnp.array,
) -> jnp.array:

    cov_scaled = cov_matrix_th_scaled + cov_matrix_exp_scaled
    std_scaled = jnp.sqrt(jnp.diag(cov_scaled))
    std = std_scaled  * std_sm_exp
    C = cov_scaled / jnp.outer(std_scaled, std_scaled)
    D = (predictions_scaled - exp_central_scaled)/std_scaled

    logpdf_rows = []
    for i in range(len(observable_indices)):
        logpdf_total = jnp.zeros_like(predictions_scaled)
        d = jnp.take(D, observable_indices[i])
        c = jnp.take(jnp.take(C, observable_indices[i], axis=0), observable_indices[i], axis=1)

        logdet_corr = jnp.linalg.slogdet(c)[1]
        logprod_std2 = 2 * jnp.sum(jnp.log(jnp.take(std, observable_indices[i])))

        logpdf = -0.5 * (
            d * jsp.linalg.cho_solve(jsp.linalg.cho_factor(c), d)
            + (logdet_corr
            + logprod_std2)/len(d)
            + jnp.log(2 * jnp.pi)
        )
        logpdf_total = logpdf_total.at[observable_indices[i]].add(logpdf)
        logpdf_rows.append(logpdf_total)
    return jnp.array(logpdf_rows)

def logL_correlated_sectors_summed(
    predictions_scaled: jnp.array,
    selector_matrix: jnp.array,
    observable_indices: List[jnp.array],
    exp_central_scaled: jnp.array,
    cov_matrix_exp_scaled: jnp.array,
    cov_matrix_th_scaled: jnp.array,
) -> jnp.array:

    cov_scaled = cov_matrix_th_scaled + cov_matrix_exp_scaled
    std_scaled = jnp.sqrt(jnp.diag(cov_scaled))
    C = cov_scaled / jnp.outer(std_scaled, std_scaled)
    D = (predictions_scaled - exp_central_scaled)/std_scaled

    logL_rows = []
    for i in range(len(observable_indices)):
        logL_total = jnp.zeros_like(predictions_scaled)
        d = jnp.take(D, observable_indices[i])
        c = jnp.take(jnp.take(C, observable_indices[i], axis=0), observable_indices[i], axis=1)
        logL = -0.5 * jnp.dot(d, jsp.linalg.cho_solve(jsp.linalg.cho_factor(c), d))
        logL_rows.append(logL)
    logL_total = jnp.array(logL_rows)
    return selector_matrix @ logL_total

def logL_correlated_sectors(
    predictions_scaled: jnp.array,
    observable_indices: List[jnp.array],
    exp_central_scaled: jnp.array,
    cov_matrix_exp_scaled: jnp.array,
    cov_matrix_th_scaled: jnp.array,
) -> jnp.array:

    cov_scaled = cov_matrix_th_scaled + cov_matrix_exp_scaled
    std_scaled = jnp.sqrt(jnp.diag(cov_scaled))
    C = cov_scaled / jnp.outer(std_scaled, std_scaled)
    D = (predictions_scaled - exp_central_scaled)/std_scaled

    logL_rows = []
    for i in range(len(observable_indices)):
        logL_total = jnp.zeros_like(predictions_scaled)
        d = jnp.take(D, observable_indices[i])
        c = jnp.take(jnp.take(C, observable_indices[i], axis=0), observable_indices[i], axis=1)
        logL = -0.5 * d * jsp.linalg.cho_solve(jsp.linalg.cho_factor(c), d)
        logL_total = logL_total.at[observable_indices[i]].add(logL)
        logL_rows.append(logL_total)
    return jnp.array(logL_rows)

def combine_normal_distributions(
        measurement_name: np.ndarray,
        observables: np.ndarray,
        observable_indices: np.ndarray,
        central_value: np.ndarray,
        standard_deviation: np.ndarray,
    ) -> Dict[str, np.ndarray]:
    '''
    Combine multiple normal distributions into a single normal distribution.

    Parameters
    ----------
    measurement_name : np.ndarray
        Names of the measurements.
    observables : np.ndarray
        Names of the observables.
    observable_indices : np.ndarray
        Indices of the observables.
    central_value : np.ndarray
        Central values of the normal distributions.
    standard_deviation : np.ndarray
        Standard deviations of the normal distributions.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the combined measurement name, observables, observable indices,
        central value, and standard deviation.

    Examples
    --------
    >>> combine_normal_distributions(
    ...     measurement_name=np.array(['measurement1', 'measurement2']),
    ...     observables=np.array(['observable1', 'observable1']),
    ...     observable_indices=np.array([3, 3]),
    ...     central_value=np.array([1.0, 2.0]),
    ...     standard_deviation=np.array([0.1, 0.2])
    ... )
    {
        'measurement_name': np.array(['measurement1, measurement2']),
        'observables': np.array(['observable1']),
        'observable_indices': np.array([3]),
        'central_value': np.array([1.2]),
        'standard_deviation': np.array([0.08944272])
    }
    '''

    if len(measurement_name) > 1:
        if len(np.unique(observables)) > 1:
            raise ValueError(f"Only distributions constraining the same observable can be combined.")
        measurement_name = np.expand_dims(', '.join(np.unique(measurement_name)), axis=0)
        observables = observables[:1]
        observable_indices = observable_indices[:1]
        weights = 1 / standard_deviation**2
        central_value = np.average(central_value, weights=weights, keepdims=True)
        standard_deviation = np.sqrt(1 / np.sum(weights, keepdims=True))
    return {
        'measurement_name': measurement_name,
        'observables': observables,
        'observable_indices': observable_indices,
        'central_value': central_value,
        'standard_deviation': standard_deviation,
    }

def get_distribution_support(
        dist_type: str,
        dist_info: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get the support of one or more distributions based on the distribution parameters.

    Parameters
    ----------
    dist_type : str
        Type of the distribution (e.g., 'NumericalDistribution', 'NormalDistribution', etc.).
    dist_info : Dict[str, np.ndarray]
        Information about the distribution, such as 'central_value', 'standard_deviation', etc.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the minimum and maximum values of the support of the distributions.

    Examples
    --------
    >>> get_distribution_support('NormalDistribution', {'central_value': np.array([0.0, 1.0]), 'standard_deviation': np.array([1.0, 2.0])})
    (array([-6., -11.]), array([6., 13.]))

    '''

    if dist_type == 'NumericalDistribution':
        xp = dist_info['x']
        return np.min(xp, axis=1), np.max(xp, axis=1)
    elif dist_type == 'NormalDistribution':
        central_value = dist_info['central_value']
        standard_deviation = dist_info['standard_deviation']
        return central_value - 6*standard_deviation, central_value + 6*standard_deviation
    elif dist_type == 'HalfNormalDistribution':
        standard_deviation = dist_info['standard_deviation']
        return np.zeros_like(standard_deviation), 6*standard_deviation
    elif dist_type == 'GammaDistributionPositive':
        a = dist_info['a']
        loc = dist_info['loc']
        scale = dist_info['scale']
        mode = np.maximum(loc + (a-1)*scale, 0)
        gamma = sp.stats.gamma(a, loc, scale)
        support_min = np.maximum(np.minimum(gamma.ppf(1e-9), mode), 0)
        support_max = gamma.ppf(1-1e-9*(1-gamma.cdf(0)))
        return support_min, support_max
    else:
        raise NotImplementedError(f"Computing the support not implemented for {dist_type}.")

def log_trapz_exp(
        log_y: np.ndarray,
        x: np.ndarray,
    ) -> np.float64:
    '''
    Compute the log of the trapezoidal integral of the exponential of log_y over x.

    Parameters
    ----------
    log_y : np.ndarray
        Logarithm of the values to be integrated.
    x : np.ndarray
        Points at which log_y is defined. It is assumed that x is uniformly spaced.

    Returns
    -------
    float
        The logarithm of the trapezoidal integral of exp(log_y) over x.

    Examples
    --------
    >>> log_y = np.array([0.1, 0.2, 0.3])
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> log_trapz_exp(log_y, x)
    0.8956461395871966
    '''
    log_dx = np.log(x[1] - x[0])  # assume uniform spacing
    log_weights = np.zeros(len(x))
    log_weights[[0,-1]] = np.log(0.5)
    return log_dx + sp.special.logsumexp(log_y + log_weights)

def combine_distributions_numerically(
        constraints: Dict[str, Dict[str, np.ndarray]],
        n_points: int = 1000,
) -> Dict[str, np.ndarray]:
    '''
    Combine multiple distributions into a single numerical distribution by summing their logpdfs on a common support.

    Parameters
    ----------
    constraints : Dict[str, Dict[str, np.ndarray]]
        A dictionary where keys are distribution types (e.g., 'NumericalDistribution', 'NormalDistribution', etc.)
        and values are dictionaries containing distribution information such as 'central_value', 'standard_deviation', etc.
    n_points : int, optional
        Number of points in the common support for the output distribution. Default is 1000.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the combined numerical distribution information, including 'measurement_name', 'observables',
        'observable_indices', 'x', 'y', and 'log_y'.

    Examples
    --------
    >>> constraints = {
    ...     'NormalDistribution': {
    ...         'measurement_name': np.array(['measurement1']),
    ...         'observables': np.array(['observable1']),
    ...         'observable_indices': np.array([0]),
    ...         'central_value': np.array([1.0]),
    ...         'standard_deviation': np.array([0.8])
    ...     },
    ...     'HalfNormalDistribution': {
    ...         'measurement_name': np.array(['measurement2', 'measurement3']),
    ...         'observables': np.array(['observable1', 'observable1']),
    ...         'observable_indices': np.array([0, 0]),
    ...         'standard_deviation': np.array([0.3, 0.4])
    ...     }
    ... }
    >>> combine_distributions(constraints, n_points=1000)
    {
        'measurement_name': np.array(['measurement1, measurement2, measurement3']),
        'observables': np.array(['observable1']),
        'observable_indices': np.array([0]),
        'x': np.array([...]),  # combined support
        'y': np.array([...]),  # combined pdf values
        'log_y': np.array([...])  # combined log pdf values
    }
    '''

    # get universal parameters for output
    dist_info = next(iter(constraints.values()))
    observables_out = dist_info['observables'][:1]
    observable_indices_out = dist_info['observable_indices'][:1]

    # get measurement names in each constraint and supports of distributions
    measurement_names = []
    supports = []
    for dist_type, dist_info in constraints.items():
        supports.append(
            get_distribution_support(dist_type, dist_info)
        )
        measurement_names.append(dist_info['measurement_name'])

    # combine measurement names for output
    measurement_name_out = np.expand_dims(', '.join(np.unique(np.concatenate(measurement_names))), axis=0)

    # common support for all distributions
    support_min = np.min(np.concatenate([s[0] for s in supports]))
    support_max = np.max(np.concatenate([s[1] for s in supports]))
    xp_out = np.linspace(support_min, support_max, n_points)

    # sum the logpdfs of all distributions on the common support
    log_fp_out = np.zeros_like(xp_out)
    for dist_type, dist_info in constraints.items():
        unique_observables = np.unique(dist_info['observables'])
        if len(unique_observables) > 1 or unique_observables[0] != observables_out[0]:
            raise ValueError(f"Only distributions constraining the same observable can be combined.")
        n_constraints = len(dist_info['observables'])
        x = np.broadcast_to(xp_out, (n_constraints, n_points)).reshape(-1)
        observable_indices = np.arange(len(x))
        selector_matrix = np.concatenate([np.eye(n_points)]*n_constraints, axis=1)
        if dist_type == 'NumericalDistribution':
            xp = dist_info['x']
            log_fp = dist_info['log_y']
            xp = np.broadcast_to(xp[:, None, :], (xp.shape[0], n_points, xp.shape[1]))
            xp = xp.reshape(-1, xp.shape[2])
            log_fp = np.broadcast_to(log_fp[:, None, :], (log_fp.shape[0], n_points, log_fp.shape[1]))
            log_fp = log_fp.reshape(-1, log_fp.shape[2])
            log_fp_out += logpdf_functions_summed[dist_type](
                x,
                selector_matrix,
                observable_indices,
                xp,
                log_fp,
            )
        elif dist_type == 'NormalDistribution':
            central_value = np.broadcast_to(dist_info['central_value'], (n_points, n_constraints)).T.reshape(-1)
            standard_deviation = np.broadcast_to(dist_info['standard_deviation'], (n_points, n_constraints)).T.reshape(-1)
            log_fp_out += logpdf_functions_summed[dist_type](
                x,
                selector_matrix,
                observable_indices,
                central_value,
                standard_deviation,
            )
        elif dist_type == 'HalfNormalDistribution':
            standard_deviation = np.broadcast_to(dist_info['standard_deviation'], (n_points, n_constraints)).T.reshape(-1)
            log_fp_out += logpdf_functions_summed[dist_type](
                x,
                selector_matrix,
                observable_indices,
                standard_deviation,
            )
        elif dist_type == 'GammaDistributionPositive':
            a = np.broadcast_to(dist_info['a'], (n_points, n_constraints)).T.reshape(-1)
            loc = np.broadcast_to(dist_info['loc'], (n_points, n_constraints)).T.reshape(-1)
            scale = np.broadcast_to(dist_info['scale'], (n_points, n_constraints)).T.reshape(-1)
            log_fp_out += logpdf_functions_summed[dist_type](
                x,
                selector_matrix,
                observable_indices,
                a,
                loc,
                scale,
            )
        else:
            raise NotImplementedError(f"Combining distributions not implemented for {dist_type}.")

    # normalize the output distribution
    log_fp_out -= log_trapz_exp(log_fp_out, xp_out)

    return {
        'measurement_name': measurement_name_out,
        'observables': observables_out,
        'observable_indices': observable_indices_out,
        'x': xp_out,
        'y': np.exp(log_fp_out),
        'log_y': log_fp_out,
    }

def get_ppf_numerical_distribution(
        xp: np.ndarray,
        fp: np.ndarray,
) -> Callable:
    '''
    Get the percent-point function (PPF) for one or more numerical distributions.

    Parameters
    ----------
    xp : np.ndarray
        Points at which the PDF is defined.
    fp : np.ndarray
        PDF values at the points xp.

    Returns
    -------
    Callable
        The PPF that can be used to compute the quantiles for given probabilities.
    '''
    if xp.ndim == 1: # single distribution
        cdf = np.concatenate([[0], np.cumsum((fp[1:] + fp[:-1]) * 0.5 * np.diff(xp))])
        cdf /= cdf[-1]
        return partial(np.interp, xp=cdf, fp=xp)
    elif xp.ndim == 2: # multiple distributions
        dx = np.diff(xp, axis=1)
        avg_fp = 0.5 * (fp[:, 1:] + fp[:, :-1])
        cdf = np.cumsum(avg_fp * dx, axis=1)
        cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=1)
        cdf /= cdf[:, [-1]]

        def batched_ppf(q: Union[float, np.ndarray]) -> np.ndarray:
            """
            Batched PPF for multiple distributions.

            Parameters
            ----------
            q : float or np.ndarray
                Lower-tail probabilities at which to compute the PPF.
                - If scalar, computes PPF for that probability across all distributions.
                - If 1D array of shape (k,), computes PPF at k probabilities for all distributions.
                - If 2D array of shape (k, m), computes PPF at k probabilities for each of the m distributions.

            Returns
            -------
            np.ndarray
                The quantiles corresponding to the input probabilities.
                - If input is scalar, returns 1D array of shape (m,)
                - Otherwise returns array of shape (k, m)
            """

            q = np.asarray(q)
            scalar_input = False
            if q.ndim == 0:  # single probability for all distributions
                q = np.full((1, cdf.shape[0]), q)
                scalar_input = True
            elif q.ndim == 1:  # a vector of probabilities for all distributions
                q = np.tile(q[None, :], (1, cdf.shape[0]))
            result = np.empty_like(q)
            for i in range(q.shape[1]):  # iterate over distributions
                result[:, i] = np.interp(q[:, i], cdf[i], xp[i])
            return result[0] if scalar_input else result
        return batched_ppf

def get_ppf_gamma_distribution_positive(
        a: np.ndarray,
        loc: np.ndarray,
        scale: np.ndarray,
) -> Callable:
    """
    Get the percent-point function (PPF) for a gamma distribution restricted to positive values.

    Parameters
    ----------
    a : np.ndarray
        Shape parameter of the gamma distribution.
    loc : np.ndarray
        Location parameter of the gamma distribution.
    scale : np.ndarray
        Scale parameter of the gamma distribution.

    Returns
    -------
    Callable
        The PPF that can be used to compute the quantiles for given probabilities.
    """
    gamma = sp.stats.gamma(a, loc, scale)
    def ppf(q):
        return gamma.ppf(q + (1-q)*gamma.cdf(0))
    return ppf

def get_mode_and_uncertainty(
        dist_type: str,
        dist_info: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get the mode and uncertainty of one or more distributions based on the distribution parameters.

    A Gaussian approximation or an upper limit based on the 95% confidence level is used, depending on the distribution type and parameters.

    In case of the upper limit, the mode is set to nan.

    Parameters
    ----------
    dist_type : str
        Type of the distribution (e.g., 'NumericalDistribution', 'NormalDistribution', etc.).
    dist_info : Dict[str, np.ndarray]
        Information about the distribution, such as 'central_value', 'standard_deviation', etc.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the mode and uncertainty of the distributions.

    Examples
    --------
    >>> get_mode_and_uncertainty('NormalDistribution', {'central_value': np.array([0.0, 1.0]), 'standard_deviation': np.array([1.0, 2.0])})
    (array([0., 1.]), array([1., 2.]))
    >>> get_mode_and_uncertainty('HalfNormalDistribution', {'standard_deviation': np.array([0.3, 0.4])})
    (array([nan, nan]), array([0.588, 0.784]))
    >>> get_mode_and_uncertainty('GammaDistributionPositive', {'a': np.array([2.0, 4.0]), 'loc': np.array([-1.0, 0.0]), 'scale': np.array([1.0, 2.0])})
    (array([nan,  6.]), array([4.11300328, 3.46410162]))
    >>> central_value = np.array([[0.0], [6.4]])
    >>> standard_deviation = np.array([[1.0], [1.2]])
    >>> xp = np.broadcast_to(np.linspace(0, 10, 10000), (2, 10000))
    >>> fp = sp.stats.norm.pdf(xp, loc=central_value, scale=standard_deviation)
    >>> get_mode_and_uncertainty('NumericalDistribution', {'x': xp, 'y': fp, 'log_y': np.log(fp)})
    (array([nan, 6.4]), array([1.96, 1.2]))
    '''
    if dist_type == 'NormalDistribution':
        mode = dist_info['central_value']
        uncertainty = dist_info['standard_deviation']
        return mode, uncertainty
    elif dist_type == 'HalfNormalDistribution':
        uncertainty = dist_info['standard_deviation']*1.96  # 95% CL
        return np.full_like(uncertainty, np.nan), uncertainty
    elif dist_type == 'GammaDistributionPositive':
        a = dist_info['a']
        loc = dist_info['loc']
        scale = dist_info['scale']
        mode = np.maximum(loc + (a-1)*scale, 0)

        # if mode is negative, use the 95% CL upper limit, otherwise use the standard deviation at the mode
        upper_limit = mode <= 0
        gaussian = ~upper_limit
        uncertainty = np.empty_like(mode, dtype=float)
        uncertainty[gaussian] = np.sqrt((loc[gaussian]-mode[gaussian])**2 / (a[gaussian]-1))  # standard deviation at the mode, defined as sqrt(-1/(d^2/dx^2 log(gamma(x, a, loc, scale))))
        ppf = get_ppf_gamma_distribution_positive(a[upper_limit], loc[upper_limit], scale[upper_limit])
        uncertainty[upper_limit] = ppf(0.95)  # 95% CL upper limit using the ppf of the gamma distribution restricted to positive values
        mode[upper_limit] = np.nan  # set the modes to nan where they are not defined

        # check if mode/uncertainty is smaller than 1.7 and mode > 0, in this case compute 95% CL upper limit
        # 1.7 is selected as threshold where the gaussian and halfnormal approximation are approximately equally good based on the KL divergence
        upper_limit = (mode/uncertainty < 1.7) & (mode > 0)
        ppf = get_ppf_gamma_distribution_positive(a[upper_limit], loc[upper_limit], scale[upper_limit])
        uncertainty[upper_limit] = ppf(0.95)  # 95% CL upper limit using the ppf of the gamma distribution restricted to positive values
        mode[upper_limit] = np.nan
        return mode, uncertainty
    elif dist_type == 'NumericalDistribution':
        xp = dist_info['x']
        log_fp = dist_info['log_y']
        fp = dist_info['y']
        n_constraints = len(log_fp)
        mode = np.empty(n_constraints, dtype=float)
        uncertainty = np.empty(n_constraints, dtype=float)
        for i in range(n_constraints):
            log_fp_i = log_fp[i]
            fp_i = fp[i]
            xp_i = xp[i]
            fit_points = log_fp_i > np.max(log_fp_i) - 0.5  # points of logpdf within 0.5 of the maximum
            a, b, _ = np.polyfit(xp_i[fit_points], log_fp_i[fit_points], 2)  # fit a quadratic polynomial to the logpdf
            mode_i = -b / (2 * a)
            uncertainty_i = np.sqrt(-1 / (2 * a))
            if np.abs(mode_i/uncertainty_i) > 1.7:  # if mode/uncertainty is larger than 1.7, use gaussian approximation
                mode[i] = mode_i
                uncertainty[i] = uncertainty_i
            else:  # compute 95% CL upper limit using ppf of the numerical distribution
                ppf = get_ppf_numerical_distribution(xp_i, fp_i)
                mode[i] = np.nan
                uncertainty[i] = ppf(0.95)
        return mode, uncertainty

def get_inverse_transform_samples(
        ppf: Callable,
        n_samples: int,
        n_constraints: int
    ) -> np.ndarray:
    """
    Generate samples from a distribution using inverse transform sampling.

    Parameters
    ----------
    ppf : Callable
        The percent-point function (PPF) of the distribution.
    n_samples : int
        The number of samples to generate.
    n_constraints : int
        The number of constraints.

    Returns
    -------
    np.ndarray
        An array of samples drawn from the distribution defined by the PPF.
    """
    return ppf(np.random.uniform(0, 1, (n_samples, n_constraints))).T

def get_distribution_samples(
        dist_type: str,
        dist_info: Dict[str, np.ndarray],
        n_samples: int,
        seed: Optional[int] = None,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Generate samples from a specified distribution type using the provided distribution information.

    Parameters
    ----------
    dist_type : str
        Type of the distribution (e.g., 'NumericalDistribution', 'NormalDistribution', etc.).
    dist_info : Dict[str, np.ndarray]
        Information about the distribution, such as 'central_value', 'standard_deviation', etc.
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    List[np.ndarray] or np.ndarray
        A list of arrays in case of MultivariateNormalDistribution, where the length of the list is the number of constraints,
        and each array is of shape (n_observables, n_samples).
        For other distributions, returns a single array of shape (n_constraints, n_samples).

    Examples
    --------
    >>> dist_info = {
    ...     'central_value': np.array([0.0, 1.0]),
    ...     'standard_deviation': np.array([1.0, 2.0])
    ... }
    >>> get_distribution_samples('NormalDistribution', dist_info, n_samples=1000)
    array([[ 0.12345678,  1.23456789, ...
           [-0.98765432,  2.34567890, ...]])
    """
    if seed is not None:
        np.random.seed(seed)
    if dist_type == 'GammaDistributionPositive':
        a = dist_info['a']
        loc = dist_info['loc']
        scale = dist_info['scale']
        n_constraints = len(a)
        ppf = get_ppf_gamma_distribution_positive(a, loc, scale)
        return get_inverse_transform_samples(ppf, n_samples, n_constraints)
    elif dist_type == 'NumericalDistribution':
        xp = dist_info['x']
        fp = dist_info['y']
        ppf = get_ppf_numerical_distribution(xp, fp)
        n_constraints = len(xp)
        return get_inverse_transform_samples(ppf, n_samples, n_constraints)
    elif dist_type == 'NormalDistribution':
        central_value = dist_info['central_value']
        standard_deviation = dist_info['standard_deviation']
        n_constraints = len(central_value)
        return np.random.normal(central_value, standard_deviation, size=(n_samples, n_constraints)).T
    elif dist_type == 'MultivariateNormalDistribution':
        central_value = dist_info['central_value']
        standard_deviation = dist_info['standard_deviation']
        inverse_correlation = dist_info['inverse_correlation']
        samples = []
        n_constraints = len(central_value)
        for i in range(n_constraints):
            correlation = np.linalg.inv(inverse_correlation[i])  # TODO: think about saving the correlation matrix also in the constraint dict
            samples.append(
                np.random.multivariate_normal(
                    central_value[i],
                    correlation * np.outer(standard_deviation[i], standard_deviation[i]), n_samples).T
            )
        return samples
    elif dist_type == 'HalfNormalDistribution':
        standard_deviation = dist_info['standard_deviation']
        n_constraints = len(standard_deviation)
        return np.abs(np.random.normal(0, standard_deviation, size=(n_samples, n_constraints)).T)
    else:
        raise ValueError(f"Sampling not implemented for distribution type: {dist_type}")
