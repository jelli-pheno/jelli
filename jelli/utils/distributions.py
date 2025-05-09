import numpy as np
from jax import jit, vmap, numpy as jnp, scipy as jsp
from functools import partial
from flavio.statistics.probability import GammaDistribution, NormalDistribution, NumericalDistribution, _convolve_numerical

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
            'central_value': loc,
        }
    return distribution_type, parameters

interp_log_pdf = partial(jnp.interp, left=LOG_ZERO, right=LOG_ZERO)

@jit
def logpdf_numerical_distribution(predictions, observable_indices, x, log_y):
    predictions = jnp.asarray(predictions)
    predictions = jnp.take(predictions, observable_indices)
    return vmap(interp_log_pdf)(predictions, x, log_y)

@jit
def logpdf_normal_distribution(predictions, observable_indices, mean, std):
    predictions = jnp.asarray(predictions)
    predictions = jnp.take(predictions, observable_indices)
    return jsp.stats.norm.logpdf(predictions, loc=mean, scale=std)

@jit
def logpdf_gamma_distribution_positive(predictions, observable_indices, a, loc, scale):
    predictions = jnp.asarray(predictions)
    predictions = jnp.take(predictions, observable_indices)
    log_pdf_scale = jnp.log(1/(1-jsp.stats.gamma.cdf(0, a, loc=loc, scale=scale)))
    positive_logpdf = jsp.stats.gamma.logpdf(
        predictions, a, loc=loc, scale=scale
    ) + log_pdf_scale
    return jnp.where(predictions>=0, positive_logpdf, LOG_ZERO)
