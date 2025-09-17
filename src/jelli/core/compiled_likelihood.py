from typing import List, Union, Tuple, Callable
from functools import partial
from jax import jit, grad, value_and_grad, hessian, numpy as jnp

class CompiledLikelihood:
    '''
    A class to retrieve JIT compiled likelihood functions for the global likelihood instance.

    Parameters
    ----------
    global_likelihood_instance : `GlobalLikelihood`
        An instance of the `GlobalLikelihood` class.
    par_list : List[Tuple[str, str]]
        A list of tuples specifying the parameters to be considered. Each tuple contains the parameter name and its type (e.g., `('param1', 'R')` for a real parameter, `('param2', 'I')` for an imaginary parameter).
    likelihood : str or Tuple[str, ...], optional
        The likelihood to be used. Default is 'global'.
    par_dep_cov : bool, optional
        If `True`, the covariance matrix depends on the parameters. Default is `False`.

    Attributes
    ----------
    global_likelihood_instance : `GlobalLikelihood`
        The instance of the `GlobalLikelihood` class.
    par_list : List[Tuple[str, str]]
        The list of parameters considered.
    likelihood : str or Tuple[str, ...]
        The likelihood used.
    par_dep_cov : bool
        Indicates if the covariance matrix depends on the parameters.
    _negative_log_likelihood_function : Callable
        The function to compute the negative log-likelihood.
    _log_likelihood_data : dict
        The data required for the log-likelihood computation.
    _functions : dict
        A cache for the compiled functions.

    Methods
    -------
    negative_log_likelihood_value(precompiled=True) -> Callable
        Get the jitted function for the negative log-likelihood value.
    negative_log_likelihood_grad(argnums=0, precompiled=True) -> Callable
        Get the jitted function for the gradient of the negative log-likelihood.
    negative_log_likelihood_value_and_grad(argnums=0, precompiled=True) -> Callable
        Get the jitted function for both the value and gradient of the negative log-likelihood.
    negative_log_likelihood_hessian(argnums=0, precompiled=True) -> Callable
        Get the jitted function for the Hessian of the negative log-likelihood.
    observed_fisher_information(argnums=0, precompiled=True) -> Callable
        Get the jitted function for the observed Fisher information (same as Hessian).
    negative_log_likelihood_inverse_hessian(argnums=0, precompiled=True) -> Callable
        Get the jitted function for the inverse of the Hessian of the negative log-likelihood.
    asymptotic_covariance(argnums=0, precompiled=True) -> Callable
        Get the jitted function for the asymptotic covariance (same as inverse Hessian).

    Examples
    --------
    Initialize the `CompiledLikelihood` class with a `GlobalLikelihood` instance and a parameter list:

    >>> gl = GlobalLikelihood(...)
    >>> par_list = [('param1', 'R'), ('param2', 'I')]
    >>> likelihood = 'global'
    >>> par_dep_cov = False
    >>> compiled_likelihood = CompiledLikelihood(gl, par_list, likelihood, par_dep_cov)

    Get the jitted function for the negative log-likelihood value:

    >>> nll_value_func = compiled_likelihood.negative_log_likelihood_value()
    >>> nll_value = nll_value_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the gradient of the negative log-likelihood with respect to the parameters:

    >>> nll_grad_func = compiled_likelihood.negative_log_likelihood_grad(argnums=0)
    >>> nll_grad = nll_grad_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the gradient of the negative log-likelihood with respect to both the parameters and the scale:

    >>> nll_grad_func = compiled_likelihood.negative_log_likelihood_grad(argnums=(0, 1))
    >>> nll_grad = nll_grad_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for both the value and gradient of the negative log-likelihood:

    >>> nll_value_and_grad_func = compiled_likelihood.negative_log_likelihood_value_and_grad(argnums=0)
    >>> nll_value, nll_grad = nll_value_and_grad_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the Hessian of the negative log-likelihood:

    >>> nll_hess_func = compiled_likelihood.negative_log_likelihood_hessian(argnums=0)
    >>> nll_hess = nll_hess_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the observed Fisher information (same as Hessian):

    >>> fisher_info_func = compiled_likelihood.observed_fisher_information(argnums=0)
    >>> fisher_info = fisher_info_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the inverse of the Hessian of the negative log-likelihood:

    >>> nll_inv_hess_func = compiled_likelihood.negative_log_likelihood_inverse_hessian(argnums=0)
    >>> nll_inv_hess = nll_inv_hess_func(jnp.array([0.1, 0.2]), 1000.0)

    Get the jitted function for the asymptotic covariance (same as inverse Hessian):

    >>> asymp_cov_func = compiled_likelihood.asymptotic_covariance(argnums=0)
    >>> asymp_cov = asymp_cov_func(jnp.array([0.1, 0.2]), 1000.0)
    '''

    def __init__(
        self,
        global_likelihood_instance,
        par_list: List[Tuple[str, str]],
        likelihood: Union[str, Tuple[str, ...]] = 'global',
        par_dep_cov: bool = False,
    ):
        '''
        Initialize the `CompiledLikelihood` class.

        Parameters
        ----------
        global_likelihood_instance : `jelli.core.global_likelihood.GlobalLikelihood`
            An instance of the `GlobalLikelihood` class.
        par_list : List[Tuple[str, str]]
            A list of tuples specifying the parameters to be considered. Each tuple contains the parameter name and its type (e.g., `('param1', 'R')` for a real parameter, `('param2', 'I')` for an imaginary parameter).
        likelihood : str or Tuple[str, ...], optional
            The likelihood to be used. Default is 'global'.
        par_dep_cov : bool, optional
            If `True`, the covariance matrix depends on the parameters. Default is `False`.

        Returns
        -------
        None

        Examples
        --------

        Initialize the `CompiledLikelihood` class with a `GlobalLikelihood` instance and a parameter list:

        >>> gl = GlobalLikelihood(...)
        >>> par_list = [('param1', 'R'), ('param2', 'I')]
        >>> likelihood = 'global'
        >>> par_dep_cov = False
        >>> compiled_likelihood = CompiledLikelihood(gl, par_list, likelihood, par_dep_cov)
        '''
        self.global_likelihood_instance = global_likelihood_instance
        self.par_list = par_list
        self.likelihood = likelihood
        self.par_dep_cov = par_dep_cov
        self._negative_log_likelihood_function, self._log_likelihood_data = self.global_likelihood_instance.get_negative_log_likelihood(par_list, likelihood, par_dep_cov)
        self._functions = {}

    def negative_log_likelihood_value(
        self,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the negative log-likelihood value.

        Parameters
        ----------
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the negative log-likelihood value.

        Examples
        --------
        Get the jitted function for the negative log-likelihood value:

        >>> nll_value_func = compiled_likelihood.negative_log_likelihood_value()
        >>> nll_value = nll_value_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        if "negative_log_likelihood_value" not in self._functions:
            f = partial(
                jit(self._negative_log_likelihood_function),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._functions["negative_log_likelihood_value"] = f
        return self._functions["negative_log_likelihood_value"]

    def negative_log_likelihood_grad(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the gradient of the negative log-likelihood.

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the gradient is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the gradient of the negative log-likelihood.

        Examples
        --------
        Get the jitted function for the gradient of the negative log-likelihood with respect to the parameters:

        >>> nll_grad_func = compiled_likelihood.negative_log_likelihood_grad(argnums=0)
        >>> nll_grad = nll_grad_func(jnp.array([0.1, 0.2]), 1000.0)

        Get the jitted function for the gradient of the negative log-likelihood with respect to both the parameters and the scale:

        >>> nll_grad_func = compiled_likelihood.negative_log_likelihood_grad(argnums=(0, 1))
        >>> nll_grad = nll_grad_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        if ("negative_log_likelihood_grad", argnums) not in self._functions:
            f = partial(
                jit(grad(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._functions[("negative_log_likelihood_grad", argnums)] = f
        return self._functions[("negative_log_likelihood_grad", argnums)]

    def negative_log_likelihood_value_and_grad(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for both the value and gradient of the negative log-likelihood.

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the gradient is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for both the value and gradient of the negative log-likelihood.

        Examples
        --------
        Get the jitted function for both the value and gradient of the negative log-likelihood:

        >>> nll_value_and_grad_func = compiled_likelihood.negative_log_likelihood_value_and_grad(argnums=0)
        >>> nll_value, nll_grad = nll_value_and_grad_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        if ("negative_log_likelihood_value_and_grad", argnums) not in self._functions:
            f = partial(
                jit(value_and_grad(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._functions[("negative_log_likelihood_value_and_grad", argnums)] = f
        return self._functions[("negative_log_likelihood_value_and_grad", argnums)]

    def negative_log_likelihood_hessian(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the Hessian of the negative log-likelihood.

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the Hessian is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the Hessian of the negative log-likelihood.

        Examples
        --------
        Get the jitted function for the Hessian of the negative log-likelihood:

        >>> nll_hess_func = compiled_likelihood.negative_log_likelihood_hessian(argnums=0)
        >>> nll_hess = nll_hess_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        if ("negative_log_likelihood_hessian", argnums) not in self._functions:
            f = partial(
                jit(hessian(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._functions[("negative_log_likelihood_hessian", argnums)] = f
        return self._functions[("negative_log_likelihood_hessian", argnums)]

    def observed_fisher_information(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the observed Fisher information (same as Hessian).

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the Fisher information is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the observed Fisher information.

        Examples
        --------
        Get the jitted function for the observed Fisher information (same as Hessian):

        >>> fisher_info_func = compiled_likelihood.observed_fisher_information(argnums=0)
        >>> fisher_info = fisher_info_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        return self.negative_log_likelihood_hessian(argnums=argnums, precompiled=precompiled)

    def negative_log_likelihood_inverse_hessian(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the inverse of the Hessian of the negative log-likelihood.

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the inverse Hessian is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the inverse of the Hessian of the negative log-likelihood.

        Examples
        --------
        Get the jitted function for the inverse of the Hessian of the negative log-likelihood:

        >>> nll_inv_hess_func = compiled_likelihood.negative_log_likelihood_inverse_hessian(argnums=0)
        >>> nll_inv_hess = nll_inv_hess_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        if ("negative_log_likelihood_inverse_hessian", argnums) not in self._functions:
            def f(par_array, scale, log_likelihood_data):
                hess = hessian(self._negative_log_likelihood_function, argnums=argnums)(par_array, scale, log_likelihood_data)
                # regularize the inverse
                d = jnp.sqrt(jnp.diag(hess))
                d2 = jnp.outer(d, d)
                R = hess / d2
                inv_R = jnp.linalg.inv(R)
                inv_hess = inv_R / d2
                return inv_hess
            f = partial(
                jit(f),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._functions[("negative_log_likelihood_inverse_hessian", argnums)] = f
        return self._functions[("negative_log_likelihood_inverse_hessian", argnums)]

    def asymptotic_covariance(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:
        '''
        Get the jitted function for the asymptotic covariance (same as inverse Hessian).

        Parameters
        ----------
        argnums : int or Tuple[int, ...], optional
            The argument numbers with respect to which the asymptotic covariance is computed. Default is `0` (the parameters). Use `(0, 1)` to include the scale as well.
        precompiled : bool, optional
            If `True`, precompile the function. Default is `True`.

        Returns
        -------
        Callable
            The jitted function for the asymptotic covariance.

        Examples
        --------
        Get the jitted function for the asymptotic covariance (same as inverse Hessian):

        >>> asymp_cov_func = compiled_likelihood.asymptotic_covariance(argnums=0)
        >>> asymp_cov = asymp_cov_func(jnp.array([0.1, 0.2]), 1000.0)
        '''
        return self.negative_log_likelihood_inverse_hessian(argnums=argnums, precompiled=precompiled)
