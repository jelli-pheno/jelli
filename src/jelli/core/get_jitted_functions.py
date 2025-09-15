from typing import List, Union, Tuple, Callable
from functools import partial
from jax import jit, grad, value_and_grad, hessian, numpy as jnp

class GetJittedFunctions:


    def __init__(
        self,
        global_likelihood_instance,
        par_list: List[Tuple[str, str]],
        likelihood: Union[str, Tuple[str, ...]] = 'global',
        par_dep_cov: bool = False,
    ):

        self.global_likelihood_instance = global_likelihood_instance
        self.par_list = par_list
        self.likelihood = likelihood
        self.par_dep_cov = par_dep_cov
        self._negative_log_likelihood_function, self._log_likelihood_data = self.global_likelihood_instance.get_negative_log_likelihood(par_list, likelihood, par_dep_cov)
        self._jitted_functions = {}

    def negative_log_likelihood_value(
        self,
        precompiled: bool = True,
    ) -> Callable:

        if "negative_log_likelihood_value" not in self._jitted_functions:
            f = partial(
                jit(self._negative_log_likelihood_function),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._jitted_functions["negative_log_likelihood_value"] = f
        return self._jitted_functions["negative_log_likelihood_value"]

    def negative_log_likelihood_grad(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        if ("negative_log_likelihood_grad", argnums) not in self._jitted_functions:
            f = partial(
                jit(grad(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._jitted_functions[("negative_log_likelihood_grad", argnums)] = f
        return self._jitted_functions[("negative_log_likelihood_grad", argnums)]

    def negative_log_likelihood_value_and_grad(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        if ("negative_log_likelihood_value_and_grad", argnums) not in self._jitted_functions:
            f = partial(
                jit(value_and_grad(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._jitted_functions[("negative_log_likelihood_value_and_grad", argnums)] = f
        return self._jitted_functions[("negative_log_likelihood_value_and_grad", argnums)]

    def negative_log_likelihood_hessian(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        if ("negative_log_likelihood_hessian", argnums) not in self._jitted_functions:
            f = partial(
                jit(hessian(self._negative_log_likelihood_function, argnums=argnums)),
                log_likelihood_data=self._log_likelihood_data,
            )
            if precompiled:
                f(jnp.zeros(len(self.par_list)), self.global_likelihood_instance._reference_scale)
            self._jitted_functions[("negative_log_likelihood_hessian", argnums)] = f
        return self._jitted_functions[("negative_log_likelihood_hessian", argnums)]

    def observed_fisher_information(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        return self.negative_log_likelihood_hessian(argnums=argnums, precompiled=precompiled)

    def negative_log_likelihood_inverse_hessian(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        if ("negative_log_likelihood_inverse_hessian", argnums) not in self._jitted_functions:
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
            self._jitted_functions[("negative_log_likelihood_inverse_hessian", argnums)] = f
        return self._jitted_functions[("negative_log_likelihood_inverse_hessian", argnums)]

    def asymptotic_covariance(
        self,
        argnums: Union[int, Tuple[int, ...]] = 0,
        precompiled: bool = True,
    ) -> Callable:

        return self.negative_log_likelihood_inverse_hessian(argnums=argnums, precompiled=precompiled)
