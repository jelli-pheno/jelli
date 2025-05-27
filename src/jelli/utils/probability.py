# temporarily copied from flavio to avoid importing the whole package
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
import math
import inspect
from collections import OrderedDict
import yaml
import re
from functools import lru_cache

def normal_logpdf(x, mu, sigma):
    """Logarithm of the PDF of the normal distribution"""
    # this turns out to be 2 orders of magnitude faster than scipy.stats.norm.logpdf
    if isinstance(x, float):
        _x = x
    else:
        _x = np.asarray(x)
    return -(_x-mu)**2/sigma**2/2 - math.log(math.sqrt(2*math.pi)*sigma)

def normal_pdf(x, mu, sigma):
    """PDF of the normal distribution"""
    # this turns out to be 2 orders of magnitude faster than scipy.stats.norm.logpdf
    if isinstance(x, float):
        _x = x
    else:
        _x = np.asarray(x)
    return np.exp(-(_x-mu)**2/sigma**2/2)/(np.sqrt(2*math.pi)*sigma)

@lru_cache(maxsize=10)
def confidence_level(nsigma):
    r"""Return the confidence level corresponding to a number of sigmas,
    i.e. the probability contained in the normal distribution between $-n\sigma$
    and $+n\sigma$.

    Example: `confidence_level(1)` returns approximately 0.68."""
    return (scipy.stats.norm.cdf(nsigma)-0.5)*2


def _camel_to_underscore(s):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def string_to_class(string):
    """Get a ProbabilityDistribution subclass from a string. This can
    either be the class name itself or a string in underscore format
    as returned from `class_to_string`."""
    try:
        return eval(string)
    except NameError:
        pass
    for c in ProbabilityDistribution.get_subclasses():
        if c.class_to_string() == string:
            return c
    raise NameError("Distribution " + string + " not found.")

class ProbabilityDistribution(object):
    """Common base class for all probability distributions"""

    def __init__(self, central_value, support):
        self.central_value = central_value
        self.support = support

    @classmethod
    def get_subclasses(cls):
        """Return all subclasses (including subclasses of subclasses)."""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    def get_central(self):
        return self.central_value

    @property
    def error_left(self):
        """Return the lower error"""
        return self.get_error_left()

    @property
    def error_right(self):
        """Return the upper error"""
        return self.get_error_right()

    @classmethod
    def class_to_string(cls):
        """Get a string name for a given ProbabilityDistribution subclass.

        This converts camel case to underscore and removes the word
        'distribution'.

        Example: class_to_string(AsymmetricNormalDistribution) returns
        'asymmetric_normal'.
        """
        name = _camel_to_underscore(cls.__name__)
        return name.replace('_distribution', '')

    def get_dict(self, distribution=False, iterate=False, arraytolist=False):
        """Get an ordered dictionary with arguments and values needed to
        the instantiate the distribution.

        Optional arguments (default to False):

        - `distribution`: add a 'distribution' key to the dictionary with the
        value being the string representation of the distribution's name
        (e.g. 'asymmetric_normal').
        - `iterate`: If ProbabilityDistribution instances are among the
        arguments (e.g. for KernelDensityEstimate), return the instance's
        get_dict instead of the instance as value.
        - `arraytolist`: convert numpy arrays to lists
        """
        args = inspect.signature(self.__class__).parameters.keys()
        d = self.__dict__
        od = OrderedDict()
        if distribution:
            od['distribution'] = self.class_to_string()
        od.update(OrderedDict((a, d[a]) for a in args))
        if iterate:
            for k in od:
                if isinstance(od[k], ProbabilityDistribution):
                    od[k] = od[k].get_dict(distribution=True)
        if arraytolist:
            for k in od:
                if isinstance(od[k], np.ndarray):
                    od[k] = od[k].tolist()
                if isinstance(od[k], list):
                    for i, x in enumerate(od[k]):
                        if isinstance(x, np.ndarray):
                            od[k][i] = od[k][i].tolist()
        for k in od:
            if isinstance(od[k], int):
                od[k] = int(od[k])
            elif isinstance(od[k], float):
                od[k] = float(od[k])
            if isinstance(od[k], list):
                for i, x in enumerate(od[k]):
                    if isinstance(x, float):
                        od[k][i] = float(od[k][i])
                    elif isinstance(x, int):
                        od[k][i] = int(od[k][i])
        return od

    def get_yaml(self, *args, **kwargs):
        """Get a YAML string representing the dictionary returned by the
        get_dict method.

        Arguments will be passed to `yaml.dump`."""
        od = self.get_dict(distribution=True, iterate=True, arraytolist=True)
        return yaml.dump(od, *args, **kwargs)

    def delta_logpdf(self, x, **kwargs):
        exclude = kwargs.get('exclude', None)
        if exclude is not None:
            d = len(self.central_value)
            cv = [self.central_value[i] for i in range(d) if i not in exclude]
        else:
            cv = self.central_value
        return self.logpdf(x, **kwargs) - self.logpdf(cv, **kwargs)

class NormalDistribution(ProbabilityDistribution):
    """Univariate normal or Gaussian distribution."""

    def __init__(self, central_value, standard_deviation):
        """Initialize the distribution.

        Parameters:

        - central_value: location (mode and mean)
        - standard_deviation: standard deviation
        """
        super().__init__(central_value,
                         support=(central_value - 6 * standard_deviation,
                                  central_value + 6 * standard_deviation))
        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be positive number")
        self.standard_deviation = standard_deviation

    def __repr__(self):
        return 'flavio.statistics.probability.NormalDistribution' + \
               '({}, {})'.format(self.central_value, self.standard_deviation)

    def get_random(self, size=None):
        return np.random.normal(self.central_value, self.standard_deviation, size)

    def logpdf(self, x):
        return normal_logpdf(x, self.central_value, self.standard_deviation)

    def pdf(self, x):
        return normal_pdf(x, self.central_value, self.standard_deviation)

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, self.central_value, self.standard_deviation)

    def ppf(self, x):
        return scipy.stats.norm.ppf(x, self.central_value, self.standard_deviation)

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        return nsigma * self.standard_deviation

    def get_error_right(self, nsigma=1, **kwargs):
        """Return the upper error"""
        return nsigma * self.standard_deviation

class NumericalDistribution(ProbabilityDistribution):
    """Univariate distribution defined in terms of numerical values for the
    PDF."""

    def __init__(self, x, y, central_value=None):
        """Initialize a 1D numerical distribution.

        Parameters:

        - `x`: x-axis values. Must be a 1D array of real values in strictly
          ascending order (but not necessarily evenly spaced)
        - `y`: PDF values. Must be a 1D array of real positive values with the
          same length as `x`
        - central_value: if None (default), will be set to the mode of the
          distribution, i.e. the x-value where y is largest (by looking up
          the input arrays, i.e. without interpolation!)
        """
        self.x = x
        self.y = y
        if central_value is not None:
            if x[0] <= central_value <= x[-1]:
                super().__init__(central_value=central_value,
                                 support=(x[0], x[-1]))
            else:
                raise ValueError("Central value must be within range provided")
        else:
            mode = x[np.argmax(y)]
            super().__init__(central_value=mode, support=(x[0], x[-1]))
        self.y_norm = y /  np.trapz(y, x=x)  # normalize PDF to 1
        self.y_norm[self.y_norm < 0] = 0
        self.pdf_interp = interp1d(x, self.y_norm,
                                        fill_value=0, bounds_error=False)
        _cdf = np.zeros(len(x))
        _cdf[1:] = np.cumsum(self.y_norm[:-1] * np.diff(x))
        _cdf = _cdf/_cdf[-1] # normalize CDF to 1
        self.ppf_interp = interp1d(_cdf, x)
        self.cdf_interp = interp1d(x, _cdf)

    def __repr__(self):
        return 'flavio.statistics.probability.NumericalDistribution' + \
               '({}, {})'.format(self.x, self.y)

    def get_random(self, size=None):
        """Draw a random number from the distribution.

        If size is not None but an integer N, return an array of N numbers."""
        r = np.random.uniform(size=size)
        return self.ppf_interp(r)

    def ppf(self, x):
        return self.ppf_interp(x)

    def cdf(self, x):
        return self.cdf_interp(x)

    def pdf(self, x):
        return self.pdf_interp(x)

    def logpdf(self, x):
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(self.pdf_interp(x))

    def _find_error_cdf(self, confidence_level):
        # find the value of the CDF at the position of the left boundary
        # of the `confidence_level`% CL range by demanding that the value
        # of the PDF is the same at the two boundaries
        def x_left(a):
            return self.ppf(a)
        def x_right(a):
            return self.ppf(a + confidence_level)
        def diff_logpdf(a):
            logpdf_x_left = self.logpdf(x_left(a))
            logpdf_x_right = self.logpdf(x_right(a))
            return logpdf_x_left - logpdf_x_right
        return scipy.optimize.brentq(diff_logpdf, 0,  1 - confidence_level-1e-6)

    def get_error_left(self, nsigma=1, method='central'):
        """Return the lower error.

        'method' should be one of:

        - 'central' for a central interval (same probability on both sides of
          the central value)
        - 'hpd' for highest posterior density, i.e. probability is larger inside
          the interval than outside
        - 'limit' for a one-sided error, i.e. a lower limit"""
        if method == 'limit':
            return self.central_value - self.ppf(1 - confidence_level(nsigma))
        cdf_central = self.cdf(self.central_value)
        err_left = self.central_value - self.ppf(cdf_central * (1 - confidence_level(nsigma)))
        if method == 'central':
            return err_left
        elif method == 'hpd':
            if self.pdf(self.central_value + self.get_error_right(method='central')) == self.pdf(self.central_value - err_left):
                return err_left
            try:
                a = self._find_error_cdf(confidence_level(nsigma))
            except ValueError:
                return np.nan
            return self.central_value - self.ppf(a)
        else:
            raise ValueError("Method " + str(method) + " unknown")

    def get_error_right(self, nsigma=1, method='central'):
        """Return the upper error

        'method' should be one of:

        - 'central' for a central interval (same probability on both sides of
          the central value)
        - 'hpd' for highest posterior density, i.e. probability is larger inside
          the interval than outside
        - 'limit' for a one-sided error, i.e. an upper limit"""
        if method == 'limit':
            return self.ppf(confidence_level(nsigma)) - self.central_value
        cdf_central = self.cdf(self.central_value)
        err_right = self.ppf(cdf_central + (1 - cdf_central) * confidence_level(nsigma)) - self.central_value
        if method == 'central':
            return err_right
        elif method == 'hpd':
            if self.pdf(self.central_value - self.get_error_left(method='central')) == self.pdf(self.central_value + err_right):
                return err_right
            try:
                a = self._find_error_cdf(confidence_level(nsigma))
            except ValueError:
                return np.nan
            return self.ppf(a + confidence_level(nsigma)) - self.central_value
        else:
            raise ValueError("Method " + str(method) + " unknown")

    @classmethod
    def from_pd(cls, pd, nsteps=1000):
        if isinstance(pd, NumericalDistribution):
            return pd
        _x = np.linspace(pd.support[0], pd.support[-1], nsteps)
        _y = np.exp(pd.logpdf(_x))
        return cls(central_value=pd.central_value, x=_x, y=_y)

class GammaDistribution(ProbabilityDistribution):
    r"""A Gamma distribution defined like the `gamma` distribution in
    `scipy.stats` (with parameters `a`, `loc`, `scale`).

    The `central_value` attribute returns the location of the mode.
    """

    def __init__(self, a, loc, scale):
        if loc > 0:
            raise ValueError("loc must be negative or zero")
        # "frozen" scipy distribution object
        self.scipy_dist = scipy.stats.gamma(a=a, loc=loc, scale=scale)
        mode = loc + (a-1)*scale
        # support extends until the CDF is roughly "6 sigma"
        support_min = min(self.scipy_dist.ppf(1e-9), mode)
        support_max = self.scipy_dist.ppf(1-1e-9)
        super().__init__(central_value=mode, # the mode
                         support=(support_min, support_max))
        self.a = a
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        return 'flavio.statistics.probability.GammaDistribution' + \
               '({}, {}, {})'.format(self.a, self.loc, self.scale)

    def get_random(self, size):
        return self.scipy_dist.rvs(size=size)

    def cdf(self, x):
        return self.scipy_dist.cdf(x)

    def ppf(self, x):
        return self.scipy_dist.ppf(x)

    def logpdf(self, x):
        return self.scipy_dist.logpdf(x)

    def _find_error_cdf(self, confidence_level):
        # find the value of the CDF at the position of the left boundary
        # of the `confidence_level`% CL range by demanding that the value
        # of the PDF is the same at the two boundaries
        def x_left(a):
            return self.ppf(a)
        def x_right(a):
            return self.ppf(a + confidence_level)
        def diff_logpdf(a):
            logpdf_x_left = self.logpdf(x_left(a))
            logpdf_x_right = self.logpdf(x_right(a))
            return logpdf_x_left - logpdf_x_right
        return scipy.optimize.brentq(diff_logpdf, 0,  1 - confidence_level-1e-6)

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        a = self._find_error_cdf(confidence_level(nsigma))
        return self.central_value - self.ppf(a)

    def get_error_right(self, nsigma=1, **kwargs):
        """Return the upper error"""
        a = self._find_error_cdf(confidence_level(nsigma))
        return self.ppf(a + confidence_level(nsigma)) - self.central_value

def _convolve_numerical(probability_distributions, nsteps=10000, central_values='same'):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, NumericalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NumericalDistribution"
    if central_values == 'same':
        central_value = probability_distributions[0].central_value  # central value of the first dist
        assert all(p.central_value == central_value for p in probability_distributions), \
            "Distrubtions must all have the same central value"
    elif central_values == 'sum':
        central_value = sum([p.central_value for p in probability_distributions])
    # differences of individual central values from combined central value
    central_diffs = [central_value - p.central_value for p in probability_distributions]

    # (shifted appropriately)
    supports = (np.array([p.support for p in probability_distributions]).T + central_diffs).T
    support = (central_value - (central_value - supports[:, 0]).sum(),
               central_value - (central_value - supports[:, 1]).sum())
    delta = (support[1] - support[0]) / (nsteps - 1)
    x = np.linspace(support[0], support[1], nsteps)
    # position of the central value
    n_x_central = math.floor((central_value - support[0]) / delta)
    y = None
    for i, pd in enumerate(probability_distributions):
        y1 = np.exp(pd.logpdf(x - central_diffs[i])) * delta
        if y is None:
            # first step
            y = y1
        else:
            # convolution
            y = scipy.signal.fftconvolve(y, y1, 'full')
            # cut out the convolved signal at the right place
            y = y[n_x_central:nsteps + n_x_central]
    return NumericalDistribution(central_value=central_value, x=x, y=y)
