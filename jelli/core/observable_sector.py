
from typing import List, Dict, Tuple, Any, Callable, Union, Optional
import json
import os
import ast
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp, jit
from itertools import chain
import scipy
from collections import OrderedDict
from pathlib import Path
from rgevolve.tools import run_and_match, get_wc_basis, get_scales, get_sector_indices, get_wc_mask, matching_sectors, efts_available, bases_available, bases_installed
from rgevolve.tools.utils import normalize
from ..functions import linear2bilinear_indices
from ..utils.jax_helpers import batched_outer_ravel
from ..utils.data_io import get_json_schema, get_observable_key
from ..utils.wc_helpers import get_wc_basis_from_wcxf, get_sector_indices_from_wcxf
from .custom_basis import CustomBasis
import warnings

class ObservableSector:
    '''
    A class to represent an observable sector.

    Parameters
    ----------
    name : str
        The name of the observable sector.
    json_data : dict
        The JSON data containing the metadata and data of the observable sector

    Attributes
    ----------
    metadata : dict
        The metadata of the observable sector.
    data : dict
        The data of the observable sector.
    observable_names : list
        The names of the observables.
    polynomial_names : list
        The names of the polynomial coefficients.
    observable_expressions : list
        The expressions of the observables in terms of the polynomial coefficients.
    observable_central : jnp.array
        The central values of the polynomial coefficients of the observables (possibly an expansion to second order in the parameters).
    observable_uncertainties : jnp.array
        The uncertainties of the polynomial coefficients of the observables (possibly an expansion to second order in the parameters).
    observable_uncertainties_SM : jnp.array
        The uncertainties of the polynomial coefficients of the observables for the SM entry.
    polynomial_central : jnp.array
        The central values of the polynomial coefficients.
    eft : str
        The EFT of the observable sector.
    basis : str
        The EFT basis of the observable sector.
    scale : float, int
        The renormalization scale of the observable sector.
    sectors : list
        The EFT sectors of the observable sector.
    parameters : list
        The parameters of the observable sector.
    keys_pars_by_sectors : list
        The keys of the parameters by sector.
    keys_pars : list
        The keys of the parameters.
    sector_indices : dict
        The indices of parameters from EFT sectors in the full parameter basis.
    evolution_matrices : dict
        The Renormalization Group evolution matrices.
    construct_par_monomials_observable : function
        The function that constructs the parameter monomials from the parameter array, for the polynomial coefficients of the observables (possibly corresponding to an expansion to second order in the parameters).
    construct_par_monomials_polynomial : function
        The function that constructs the parameter monomials from the parameter array, for the polynomial coefficients.
    observable_expression_functions : list
        The functions that evaluate the observable expressions in terms of the polynomial predictions.
    prediction : function
        The function that makes a prediction for the observable sector.

    Methods
    -------
    get_prediction_data(eft, basis)
        Get the data needed to make a prediction for a given EFT and basis.
    _get_evolution_matrices(eft, basis)
        Get the Renormalization Group evolution matrices for a given EFT and basis.
    _get_prediction_function()
        Get the function that makes a prediction for the observable sector.
    _get_construct_par_monomials(keys_coeff)
        Get the function that constructs the parameter monomials from the parameter array.
    _get_observable_expression_function(i)
        Get the function that evaluates a given observable expression in terms of the polynomial predictions.
    get_class_prediction_data(eft, basis, observable_sector_names)
        Get the data needed to make a prediction for a list of observable sectors.
    get_class_prediction_function(observable_sector_names)
        Get the function that makes a prediction for a list of observable sectors.
    get_all_names(eft)
        Get the names of all observable sectors.
    get(name)
        Get an observable sector by name.
    get_all()
        Get all observable sectors.

    Examples
    --------
    Initialize an observable sector:

    >>> ObservableSector(json_data)

    Load an observable sector from a json file:

    >>> ObservableSector.load('./observable_sector.json')

    Load all observable sectors from a directory containing json files:

    >>> ObservableSector.load('./observable_sectors/')

    Get the prediction data for the observable sector:

    >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')

    Make a prediction for the observable sector:

    >>> par_array = jnp.array(np.random.rand(2499))*1e-7
    >>> scale = 1000
    >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')
    >>> prediction(par_array, scale, prediction_data)

    Get an observable sector by name:

    >>> ObservableSector.get('observable_sector_1')

    Get all observable sectors:

    >>> ObservableSector.get_all()

    Get the names of all observable sectors:
    >>> ObservableSector.get_all_names()

    Get the names of all observable sectors that can provide predictions in the 'SMEFT' basis 'Warsaw':
    >>> ObservableSector.get_all_names('SMEFT', 'Warsaw')

    Get the names of all observable sectors that can provide predictions in the 'WET' basis 'flavio':
    >>> ObservableSector.get_all_names('WET', 'flavio')

    Get the names of all observable sectors that can provide predictions in the custom basis 'custom_basis':
    >>> ObservableSector.get_all_names(custom_basis='custom_basis')

    Get the prediction data for a list of observable sectors:

    >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])

    Get the prediction function for a list of observable sectors:

    >>> prediction = ObservableSector.get_class_prediction_function(['observable_sector_1', 'observable_sector_2'])

    Make a prediction for a list of observable sectors:

    >>> par_array = jnp.array(np.random.rand(2499))*1e-7
    >>> scale = 1000
    >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])
    >>> prediction = ObservableSector.get_class_prediction_function(['observable_sector_1', 'observable_sector_2'])
    >>> prediction(par_array, scale, prediction_data)
    '''

    _observable_sectors: Dict[str, 'ObservableSector'] = {}  # Class attribute to store all observable sectors
    _popxf_versions = ['1.0'] # List of supported versions of the popxf JSON schema

    def __init__(self, name: str,  json_data: Dict[str, Any])-> None:
        '''
        Initialize an observable sector.

        Parameters
        ----------
        name : str
            The name of the observable sector.
        json_data : dict
            The JSON data containing the metadata and data of the observable sector.

        Returns
        -------
        None

        Examples
        --------
        Initialize an observable sector:

        >>> ObservableSector(json_data)
        '''
        self.name = name
        self.metadata = json_data['metadata']
        self.data = {
            k: {ast.literal_eval(kk): vv for kk, vv in v.items()}
            for k, v in json_data['data'].items()
        }

        self.observable_names = [
            get_observable_key(obs) for obs in self.metadata['observable_names']
        ]
        self.polynomial_names = self.metadata.get('polynomial_names', None)
        self.observable_expressions = self.metadata.get('observable_expressions', None)

        observable_central = self.data.get('observable_central')
        self.keys_coeff_observable = sorted(observable_central.keys())
        self.observable_central = jnp.array(np.array([observable_central[k] for k in self.keys_coeff_observable]))

        observable_uncertainties = self.data.get('observable_uncertainties', None)
        self.observable_uncertainties = jnp.array(np.array([observable_uncertainties[k] for k in self.keys_coeff_observable])) if observable_uncertainties else None
        self.observable_uncertainties_SM = self.observable_uncertainties[0].copy() if observable_uncertainties else None

        polynomial_central = self.data.get('polynomial_central', None)
        self.keys_coeff_polynomial = sorted(polynomial_central.keys()) if polynomial_central else None
        self.polynomial_central = jnp.array(np.array([polynomial_central[k] for k in self.keys_coeff_polynomial])) if self.keys_coeff_polynomial else None

        self.parameters = self.metadata['parameters']
        self.scale = self.metadata['scale']
        if isinstance(self.scale, list):
            raise NotImplementedError(
                f"The current version of ObservableSector does not support lists of scales.\n"
                f"Please use a single scale for {self.name}.\n"
            )
        wcxf = self.metadata['basis'].get('wcxf')
        if wcxf:
            self.eft = wcxf['eft']
            self.basis = wcxf['basis']
            self.custom_basis = None
            self.sectors = sorted(chain.from_iterable(
                supersectors.get(sector, [sector])
                for sector in wcxf['sectors']
                ))
            if self.basis in bases_installed.get(self.eft, []):
                self.basis_mode = 'rgevolve'
                _parameter_basis = {sector: get_wc_basis(eft=self.eft, basis=self.basis, sector=sector)
                                    for sector in self.sectors}
            else:
                self.basis_mode = 'wcxf'
                _parameter_basis = {sector: get_wc_basis_from_wcxf(eft=self.eft, basis=self.basis, sector=sector)
                                    for sector in self.sectors}
                if self.basis in bases_available.get(self.eft, []):
                    warnings.warn(
                        f"\nRG evolution matrices for the {self.basis} basis in {self.eft} are not installed. "
                        f"Falling back to fixed-scale mode: predictions will only be available at the fixed scale of {self.scale} GeV. "
                        f"You can enable RG evolution by installing the corresponding module:\n"
                        f"    pip install rgevolve.{normalize(self.eft)}.{normalize(self.basis)}",
                        UserWarning,
                        stacklevel=2
                        )
                else:
                    warnings.warn(
                        f"\nRG evolution matrices for the {self.basis} basis in {self.eft} are not available. "
                        f"Falling back to fixed-scale mode: predictions will only be available at the fixed scale of {self.scale} GeV.",
                        UserWarning,
                        stacklevel=2
                        )
            self.keys_pars_by_sectors = [
                tuple(
                    par_name
                    for par_name in _parameter_basis[sector]
                    if par_name[0] in self.parameters
                ) for sector in self.sectors
                ]

            if self.basis_mode == 'rgevolve':
                self.sector_indices = {
                    eft: {
                        basis: get_sector_indices(
                            eft, basis,
                            sectors = (
                                sorted({matching_sectors[sector] for sector in self.sectors})
                                if eft == 'SMEFT' and self.eft != 'SMEFT' else self.sectors
                            )
                        )
                        for basis in bases_installed.get(eft, [])
                    } for eft in efts_available.get(self.eft, [])
                }
                self.evolution_matrices = {
                    eft: {
                        basis: self._get_evolution_matrices(eft, basis)
                        for basis in bases_installed.get(eft, [])
                    } for eft in efts_available.get(self.eft, [])
                }
            else:
                self.sector_indices = {
                    self.eft: {
                        self.basis: get_sector_indices_from_wcxf(
                            self.eft, self.basis, self.sectors
                        )
                    }
                }
                shapes_in = [len(get_wc_basis_from_wcxf(self.eft, self.basis, sector)) for sector in self.sectors]
                shapes_out = [len(keys_pars) for keys_pars in self.keys_pars_by_sectors]
                self.evolution_matrices = {
                    self.eft: {
                        self.basis: self._get_unit_evolution_matrices(
                            shapes_in, shapes_out, 1
                        )
                    }
                }
        else:
            name = self.metadata['basis']['custom']['name']
            custom_basis = CustomBasis.get(name)
            if custom_basis:
                self.eft = None
                self.basis = None
                self.custom_basis = name
                self.sectors = [None]
                _parameter_basis = custom_basis.get_parameter_basis()
                self.basis_mode = 'custom'
                self.keys_pars_by_sectors = [
                    tuple(
                        parameter_name
                        for parameter_name in _parameter_basis
                        if parameter_name[0] in self.parameters
                    )
                    ]
                self.sector_indices = {
                    None: {
                        None: np.arange(len(_parameter_basis))
                    }
                }
                shapes_in = [len(_parameter_basis)]
                shapes_out = [len(self.keys_pars_by_sectors[0])]
                self.evolution_matrices = {
                    None: {
                        None: self._get_unit_evolution_matrices(
                            shapes_in, shapes_out, 1
                        )
                    }
                }
                warnings.warn(
                    f"\nRG evolution matrices for the custom basis {self.custom_basis} are not available. "
                    f"Falling back to fixed-scale mode: predictions will only be available at the fixed scale of {self.scale} GeV.",
                    UserWarning,
                    stacklevel=2
                )
            else:
                raise ValueError(
                    f"Basis {name} not found in CustomBasis. Please define it using `CustomBasis({name}, parameters)`."
                )

        self.keys_pars = [('', 'R')] + list(chain.from_iterable(self.keys_pars_by_sectors))

        self.construct_par_monomials_observable = self._get_construct_par_monomials(self.keys_coeff_observable)
        self.construct_par_monomials_polynomial = self._get_construct_par_monomials(self.keys_coeff_polynomial)

        self.observable_expression_functions = [
            self._get_observable_expression_function(i)
            for i in range(len(self.observable_names))
        ] if self.observable_expressions else None

        self.prediction = self._get_prediction_function()

        # Add observable sector to `_observable_sectors` class attribute
        self._observable_sectors[self.name] = self

    @classmethod
    def load(cls, path: str) -> None:
        '''
        Load an observable sector from a json file or several observable sectors from a directory containing json files.

        Parameters
        ----------
        path : str
            Path to a json file or a directory containing json files.

        Returns
        -------
        None

        Examples
        --------
        Load an observable sector from a json file:

        >>> ObservableSector.load('./observable_sector.json')

        Load all observable sectors from a directory containing json files:

        >>> ObservableSector.load('./observable_sectors/')
        '''
        # load all json files in the directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.json'):
                    cls._load_file(os.path.join(path, file))
        # load single json file
        else:
            cls._load_file(path)

    @classmethod
    def _load_file(cls, path: str) -> None:
        '''
        Load an observable sector from a json file.

        Parameters
        ----------
        path : str
            Path to a json file.

        Returns
        -------
        None
        '''
        path = Path(path)
        with path.open('r') as f:
            json_data = json.load(f)
        schema_name, schema_version = get_json_schema(json_data)
        if schema_name == 'popxf' and schema_version in cls._popxf_versions:
            cls(path.stem, json_data)


    def get_prediction_data(self, eft: str, basis: str) -> List[jnp.array]:
        '''
        Get the data needed to make a prediction for a given EFT and basis.

        Parameters
        ----------
        eft : str
            The EFT to make the prediction in.
        basis : str
            The basis to make the prediction in.

        Returns
        -------
        list
            A list containing the sector indices, evolution matrices, evolution scales, and polynomial coefficients.

        Examples
        --------
        Get the data needed to make a prediction in the Warsaw basis of the SMEFT:

        >>> observable_sector.get_prediction_data('SMEFT', 'Warsaw')

        Get the data needed to make a prediction in the flavio basis of the WET:

        >>> observable_sector.get_prediction_data('WET', 'flavio')
        '''
        return [
            jnp.array(self.sector_indices[eft][basis]),
            jnp.array(self.evolution_matrices[eft][basis], dtype=jnp.float32),
            jnp.concatenate([
                jnp.array(self.get_scales(eft, basis), dtype=jnp.float32),
                jnp.array([jnp.nan], dtype=jnp.float32) # Add NaN to handle out-of-range cases
            ]),
            jnp.array(
                self.observable_central if self.observable_expressions is None
                else self.polynomial_central,
                dtype=jnp.float64
            ),
        ]

    def get_scales(self, eft: str, basis: str) -> np.ndarray:
        '''
        Get the scales at which the Renormalization Group evolution matrices are defined for a given EFT and basis.

        Parameters
        ----------
        eft : str
            The EFT to get the Renormalization Group scales for.
        basis : str
            The basis to get the Renormalization Group scales for.

        Returns
        -------
        np.ndarray
            The scales at which the Renormalization Group evolution matrices are defined.
        '''
        if self.basis_mode == 'rgevolve':
            return get_scales(eft, basis)
        else:
            return np.array([self.scale])

    def _get_evolution_matrices(self, eft: str, basis: str) -> np.ndarray:
        '''
        Get the Renormalization Group evolution matrices for a given EFT and basis.

        Parameters
        ----------
        eft : str
            The EFT to get the Renormalization Group evolution matrices for.
        basis : str
            The basis to get the Renormalization Group evolution matrices for.

        Returns
        -------
        np.ndarray
            The Renormalization Group evolution matrices.
        '''
        pars_out = dict(zip(self.sectors, self.keys_pars_by_sectors))
        scales_in = get_scales(eft, basis)
        if eft == 'SMEFT' and self.eft != 'SMEFT':
            sectors_in = sorted({matching_sectors[sector] for sector in self.sectors})
            shapes_in = [len(get_wc_basis(eft, basis, sector)) for sector in sectors_in]
            shapes_out = [len(keys_pars) for keys_pars in self.keys_pars_by_sectors]
        matrices_scales = []
        for scale_in in scales_in:
            matrices_sectors = []
            for sector_out in self.sectors:
                matrix_sector = run_and_match(
                    eft_in=eft, eft_out=self.eft,
                    basis_in=basis, basis_out=self.basis,
                    scale_in=scale_in, scale_out=self.scale,
                    sector_out=sector_out,
                )
                par_mask = get_wc_mask(self.eft, self.basis, sector_out, pars_out[sector_out])
                matrices_sectors.append(matrix_sector[par_mask])
            if eft == 'SMEFT' and self.eft != 'SMEFT':
                matrix_scale = np.block([
                    [
                        matrices_sectors[i]
                        if matching_sectors.get(self.sectors[i]) == sectors_in[j]
                        else np.zeros((shapes_out[i], shapes_in[j]))
                        for j in range(len(sectors_in))
                    ]
                    for i in range(len(self.sectors))
                ])
            else:
                matrix_scale = scipy.linalg.block_diag(*matrices_sectors)
            matrices_scales.append(matrix_scale)
        return np.array(matrices_scales)

    def _get_unit_evolution_matrices(self, shapes_in: List[int], shapes_out: List[int], n: int) -> np.ndarray:
        '''
        Constructs a list of block-diagonal matrices composed of identity matrices.
        Each identity matrix has shape (shapes_out[i], shapes_in[i]).

        Parameters
        ----------
        shapes_in : list
            List of input dimensions for each block.
        shapes_out : list
            List of output dimensions for each block.
        n : int
            Number of matrices to create.

        Returns:
        np.ndarray
            An array of shape (n, ..., ...) containing block-diagonal matrices.

        '''
        matrices = []
        for _ in range(n):
            blocks = [np.eye(out_dim, in_dim) for in_dim, out_dim in zip(shapes_in, shapes_out)]
            matrices.append(scipy.linalg.block_diag(*blocks))
        return np.array(matrices)

    def _get_prediction_function(self) -> Callable[
        [jnp.ndarray, Union[float, int, jnp.ndarray], List[jnp.ndarray]],
        Tuple[jnp.ndarray, jnp.ndarray],
    ]:
        '''
        Get the function that makes a prediction for the observable sector.

        Returns
        -------
        function
            The function that makes a prediction for the observable sector.

            Parameters
            ----------
            par_array : jnp.ndarray
                The parameter array. The last dimension corresponds to the parameters. The other dimensions are batch dimensions.

            scale : float, int, or jnp.ndarray
                The scale at which to make the prediction. If 'par_array' has batch dimensions, 'scale' can be a scalar or an array with the same shape as the batch dimensions of 'par_array'. If 'par_array' has no batch dimensions, 'scale' can be a scalar or an array.

            prediction_data : list
                A list containing the sector indices, evolution matrices, evolution scales, and polynomial coefficients.

            Returns
            -------
            jnp.ndarray
                The observable predictions.

            jnp.ndarray
                The parameter monomials corresponding to the `observable_central` polynomial coefficients. These are used to compute the parameter dependence of the theoretical uncertainties.

        Examples
        --------
        Get the prediction function for the observable sector:

        >>> prediction = observable_sector.prediction

        Make a prediction for the observable sector:

        >>> par_array = jnp.array(np.random.rand(2499))*1e-7
        >>> scale = 1000
        >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for the observable sector with batch dimensions in 'par_array' and a scalar 'scale':

        >>> par_array = jnp.array(np.random.rand(2, 5, 2499))*1e-7
        >>> scale = 1000
        >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for the observable sector with no batch dimensions in 'par_array' and an array 'scale':

        >>> par_array = jnp.array(np.random.rand(2499))*1e-7
        >>> scale = jnp.array([1000, 2000])
        >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for the observable sector with both 'par_array' and 'scale' having batch dimensions:

        >>> par_array = jnp.array(np.random.rand(3, 2499))*1e-7
        >>> scale = jnp.array([1000, 2000, 3000])
        >>> prediction_data = observable_sector.get_prediction_data('SMEFT', 'Warsaw')
        >>> prediction(par_array, scale, prediction_data)
        '''
        if self.observable_expressions is None:

            # Define the prediction function for the case where there are no observable expressions
            # In this case, the prediction is just the polynomial prediction
            @jit
            def prediction(
                par_array: jnp.array, scale: Union[float, int, jnp.array],
                prediction_data: List[jnp.array]
            ) -> Tuple[jnp.array, jnp.array]:
                sector_indices, evolution_matrices, evolution_scales, polynomial_coefficients = prediction_data
                par_array_sector = jnp.take(par_array, sector_indices, axis=-1)
                par_array_evolved = interpolate_rg_evolution(
                    par_array_sector, scale, evolution_matrices, evolution_scales
                )
                par_monomials = self.construct_par_monomials_observable(par_array_evolved)
                polynomial_predictions = jnp.dot(par_monomials, polynomial_coefficients)
                return polynomial_predictions, par_monomials

        else:

            # Define the prediction function for the case where there are observable expressions
            # In this case, the prediction is the observable prediction evaluated in terms of the polynomial prediction
            @jit
            def prediction(
                par_array: jnp.array, scale: Union[float, int, jnp.array],
                prediction_data: List[jnp.array]
            ) -> Tuple[jnp.array, jnp.array]:
                sector_indices, evolution_matrices, evolution_scales, polynomial_coefficients = prediction_data
                par_array_sector = jnp.take(par_array, sector_indices, axis=-1)
                par_array_evolved = interpolate_rg_evolution(
                    par_array_sector, scale, evolution_matrices, evolution_scales
                )
                par_monomials = self.construct_par_monomials_polynomial(par_array_evolved)
                par_monomials_expansion = self.construct_par_monomials_observable(par_array_evolved)
                polynomial_predictions = jnp.dot(par_monomials, polynomial_coefficients)
                observable_predictions = jnp.asarray([
                    observable_expression_function(polynomial_predictions)
                    for observable_expression_function in self.observable_expression_functions
                ])
                return jnp.moveaxis(observable_predictions, 0, -1), par_monomials_expansion

        return prediction

    def _get_construct_par_monomials(self, keys_coeff: List[tuple]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        '''
        Get the function that constructs the parameter monomials from the parameter array.

        Parameters
        ----------
        keys_coeff : list
            The keys of the polynomial coefficients.

        Returns
        -------
        function
            The function that constructs the parameter monomials from the parameter array.

            Parameters
            ----------
            par_array : jnp.ndarray
                The parameter array.

            Returns
            -------
            jnp.ndarray
                The parameter monomials.
        '''
        if not keys_coeff:
            return None
        # TODO: change name of linear2bilinear_indices to get_par_monomial_indices
        par_monomial_indices = linear2bilinear_indices(self.keys_pars, keys_coeff)

        def construct_par_monomials(par_array: jnp.ndarray) -> jnp.ndarray:

            # insert 1 (in a batch-friendly way) to account for SM term
            ones_column = jnp.ones((*par_array.shape[:-1], 1))
            par_array = jnp.concatenate([ones_column, par_array], axis=-1)

            par_monomial = batched_outer_ravel(par_array)
            return jnp.take(par_monomial, par_monomial_indices, axis=-1)
        return construct_par_monomials

    def _get_observable_expression_function(self, i: int) -> Callable[[jnp.ndarray], Union[float, jnp.ndarray]]:
        '''
        Get the function that evaluates a given observable expression in terms of the polynomial predictions.

        Parameters
        ----------
        i : int
            The index of the observable expression.

        Returns
        -------
        function
            The function that evaluates the observable expression in terms of the polynomial predictions.

            Parameters
            ----------
            polynomial_predictions : jnp.ndarray
                The polynomial predictions.

            Returns
            -------
            float or jnp.ndarray
                The value of the observable expression evaluated in terms of the polynomial predictions.
        '''

        # Create a function from the observable expression string
        s = (
            'from jax.numpy import sqrt\n'
            'def observable_expression(terms):\n'
            '    {}, = terms\n'
            '    return {}'
        ).format(
            ', '.join(self.observable_expressions[i]['terms'].keys()),
            self.observable_expressions[i]['expression'],
        )
        namespace = OrderedDict()
        exec(s, namespace)
        observable_expression = namespace.popitem()[1]

        # Create the observable expression function that takes the polynomial predictions as input
        polynomial_indices = jnp.array([
            self.polynomial_names.index(v)
            for v in self.observable_expressions[i]['terms'].values()
        ])
        def observable_expression_function(polynomial_predictions: jnp.ndarray) -> Union[float, jnp.ndarray]:
            selected_polynomial_predictions = jnp.take(polynomial_predictions, polynomial_indices, axis=-1)

            # Move the last dimension to the first dimension to allow for unpacking of the terms in batch mode
            return observable_expression(jnp.moveaxis(selected_polynomial_predictions, -1, 0))

        return observable_expression_function

    @classmethod
    def get_class_prediction_data(cls, eft: str, basis: str, observable_sector_names: List[str]) -> List[jnp.array]:
        '''
        Get the data needed to make a prediction for a list of observable sectors.

        Parameters
        ----------
        eft : str
            The EFT to make the prediction in.
        basis : str
            The basis to make the prediction in.
        observable_sector_names : list
            The names of the observable sectors to make the prediction for.

        Returns
        -------
        list
            A list of lists containing the sector indices, evolution matrices, evolution scales, and polynomial coefficients for each observable sector.

        Examples
        --------
        Get the data needed to make a prediction in the Warsaw basis of the SMEFT for a list of observable sectors:

        >>> ObservableSector.get_all_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])

        Get the data needed to make a prediction in the flavio basis of the WET for a list of observable sectors:

        >>> ObservableSector.get_all_prediction_data('WET', 'flavio', ['observable_sector_1', 'observable_sector_2'])
        '''
        return [
            cls._observable_sectors[name].get_prediction_data(eft, basis)
            for name in observable_sector_names
        ]

    @classmethod
    def get_class_prediction_function(cls, observable_sector_names: List[str]) -> Callable[
        [jnp.ndarray, Union[float, int, jnp.ndarray], List[jnp.ndarray]],
        jnp.ndarray
    ]:
        '''
        Get the function that makes a prediction for a list of observable sectors.

        Parameters
        ----------
        observable_sector_names : list
            The names of the observable sectors to make the prediction for.

        Returns
        -------
        function
            The function that makes a prediction for a list of observable sectors.

            Parameters
            ----------
            par_array : jnp.ndarray
                The parameter array. The last dimension corresponds to the parameters. The other dimensions are batch dimensions.

            scale : float, int, or jnp.ndarray
                The scale at which to make the prediction. If 'par_array' has batch dimensions, 'scale' can be a scalar or an array with the same shape as the batch dimensions of 'par_array'. If 'par_array' has no batch dimensions, 'scale' can be a scalar or an array.

            prediction_data : list
                A list of lists containing the sector indices, evolution matrices, evolution scales, and polynomial coefficients for each observable sector.

            Returns
            -------
            jnp.ndarray
                The observable predictions.

        Examples
        --------
        Get the prediction function for a list of observable sectors:

        >>> prediction = ObservableSector.get_class_prediction_function(['observable_sector_1', 'observable_sector_2'])

        Make a prediction for a list of observable sectors:

        >>> par_array = jnp.array(np.random.rand(2499))*1e-7
        >>> scale = 1000
        >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for a list of observable sectors with batch dimensions in 'par_array' and a scalar 'scale':

        >>> par_array = jnp.array(np.random.rand(2, 5, 2499))*1e-7
        >>> scale = 1000
        >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for a list of observable sectors with no batch dimensions in 'par_array' and an array 'scale':

        >>> par_array = jnp.array(np.random.rand(2499))*1e-7
        >>> scale = jnp.array([1000, 2000])
        >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])
        >>> prediction(par_array, scale, prediction_data)

        Make a prediction for a list of observable sectors with both 'par_array' and 'scale' having batch dimensions:

        >>> par_array = jnp.array(np.random.rand(3, 2499))*1e-7
        >>> scale = jnp.array([1000, 2000, 3000])
        >>> prediction_data = ObservableSector.get_class_prediction_data('SMEFT', 'Warsaw', ['observable_sector_1', 'observable_sector_2'])
        >>> prediction(par_array, scale, prediction_data)
        '''
        prediction_functions = [
            cls._observable_sectors[name].prediction
            for name in observable_sector_names
        ]

        @jit
        def prediction(
            par_array: jnp.array, scale: Union[float, int, jnp.array],
            prediction_data: List[List[jnp.array]]
        ) -> jnp.array:
            return jnp.concatenate([
                prediction_function(par_array, scale, data)[0]
                for prediction_function, data in zip(prediction_functions, prediction_data)
            ], axis=-1)

        return prediction

    @classmethod
    def get_all_names(cls, eft: Optional[str] = None, basis: Optional[str] = None, custom_basis: Optional[str]=None) -> List[str]:
        '''
        Get the names of all observable sectors.

        Parameters
        ----------
        eft : str, optional
            The EFT for which the observable sectors can provide predictions.
        basis : str, optional
            The basis for which the observable sectors can provide predictions
        custom_basis : str, optional
            The custom basis for which the observable sectors can provide predictions.

        If all parameters are None, all observable sectors are returned.

        Returns
        -------
        list
            The names of all observable sectors.

        Examples
        --------
        Get the names of all observable sectors:
        >>> ObservableSector.get_all_names()

        Get the names of all observable sectors that can provide predictions in the 'SMEFT' basis 'Warsaw':
        >>> ObservableSector.get_all_names('SMEFT', 'Warsaw')

        Get the names of all observable sectors that can provide predictions in the 'WET' basis 'flavio':
        >>> ObservableSector.get_all_names('WET', 'flavio')

        Get the names of all observable sectors that can provide predictions in the custom basis 'custom_basis':
        >>> ObservableSector.get_all_names(custom_basis='custom_basis')
        '''

        if custom_basis is not None:
            if eft is not None or basis is not None:
                raise ValueError(
                    'The custom_basis parameter cannot be used together with the eft or basis parameters.'
                )
            return sorted(
                name
                for name, observable_sector in cls._observable_sectors.items()
                if observable_sector.custom_basis == custom_basis
            )
        elif eft is not None:
            if basis is not None:
                observable_sectors_wcxf = sorted(name for name, observable_sector in cls._observable_sectors.items()
                                                 if observable_sector.basis_mode == 'wcxf'
                                                 and observable_sector.basis == basis
                                                 and observable_sector.eft == eft)
                if observable_sectors_wcxf:
                    return observable_sectors_wcxf
                else:
                    if basis in bases_installed.get(eft, []):
                        return sorted(
                            name
                            for name, observable_sector in cls._observable_sectors.items()
                            if observable_sector.basis_mode == 'rgevolve'
                            and eft in efts_available.get(observable_sector.eft, [])
                            )
                    else:
                        return []
            else:
                raise ValueError(
                    'The basis parameter is required when the eft parameter is provided.'
                )
        elif basis is not None:
            raise ValueError(
                'The basis parameter is only valid when the eft parameter is also provided.'
            )
        else:
            return sorted(cls._observable_sectors.keys())

    @classmethod
    def get(cls, name: str) -> 'ObservableSector':
        '''
        Get an observable sector by name.

        Parameters
        ----------
        name : str
            The name of the observable sector.

        Returns
        -------
        ObservableSector
            The observable sector.

        Examples
        --------
        Get an observable sector by name:

        >>> ObservableSector.get('observable_sector_1')
        '''
        return cls._observable_sectors[name]

    @classmethod
    def get_all(cls) -> List['ObservableSector']:
        '''
        Get all observable sectors.

        Returns
        -------
        list
            A list of all observable sectors.

        Examples
        --------
        Get all observable sectors:

        >>> ObservableSector.get_all()
        '''
        return [cls._observable_sectors[name] for name in cls.get_all_names()]

def interpolate_rg_evolution(
    par_array: jnp.ndarray,
    scale: Union[float, int, jnp.ndarray],
    evolution_matrices: jnp.ndarray,
    evolution_scales: jnp.ndarray
) -> jnp.ndarray:
    '''
    Interpolate the Renormalization Group evolution of the parameters.

    Parameters
    ----------
    par_array : jnp.ndarray
        The parameter array. The last dimension corresponds to the parameters. The other dimensions are batch dimensions.

    scale : float, int or jnp.ndarray
        The scale at which to make the prediction. If 'par_array' has batch dimensions, 'scale' can be a scalar or an array with the same shape as the batch dimensions of 'par_array'. If 'par_array' has no batch dimensions, 'scale' can be a scalar or an array.

    evolution_matrices : jnp.ndarray
        The Renormalization Group evolution matrices.

    evolution_scales : jnp.ndarray
        The scales at which the Renormalization Group evolution matrices are defined.

    Returns
    -------
    jnp.ndarray
        The evolved parameter array.

    Examples
    --------
    Interpolate the Renormalization Group evolution of the parameters:

    >>> par_array = jnp.array(np.random.rand(2499))*1e-7
    >>> scale = 1000
    >>> interpolate_rg_evolution(par_array, scale, evolution_matrices, evolution_scales)

    Interpolate the Renormalization Group evolution of the parameters with batch dimensions in 'par_array' and a scalar 'scale':

    >>> par_array = jnp.array(np.random.rand(2, 5, 2499))*1e-7
    >>> scale = 1000
    >>> interpolate_rg_evolution(par_array, scale, evolution_matrices, evolution_scales)

    Interpolate the Renormalization Group evolution of the parameters with no batch dimensions in 'par_array' and an array 'scale':

    >>> par_array = jnp.array(np.random.rand(2499))*1e-7
    >>> scale = jnp.array([1000, 2000])
    >>> interpolate_rg_evolution(par_array, scale, evolution_matrices, evolution_scales)

    Interpolate the Renormalization Group evolution of the parameters with both 'par_array' and 'scale' having batch dimensions:

    >>> par_array = jnp.array(np.random.rand(3, 2499))*1e-7
    >>> scale = jnp.array([1000, 2000, 3000])
    >>> interpolate_rg_evolution(par_array, scale, evolution_matrices, evolution_scales)
    '''

    # Searchsorted logic (supports batched scale)
    index_low = jnp.searchsorted(evolution_scales, scale, side='right') - 1
    index_high = index_low + 1

    # Extract scales and matrices (supports batched indices)
    scale_low = jnp.take(evolution_scales, index_low)
    scale_high = jnp.take(evolution_scales, index_high)
    matrix_low = jnp.take(evolution_matrices, index_low, axis=0)
    matrix_high = jnp.take(evolution_matrices, index_high, axis=0)

    # Expand scales for broadcasting with matrices
    scale = jnp.expand_dims(scale, axis=(-2, -1))
    scale_low = jnp.expand_dims(scale_low, axis=(-2, -1))
    scale_high = jnp.expand_dims(scale_high, axis=(-2, -1))

    # Logarithmic interpolation
    matrix = jnp.where(
        scale == scale_low,
        matrix_low,
        matrix_low + (matrix_high - matrix_low) * jnp.log(scale / scale_low) / jnp.log(scale_high / scale_low)
    )

    # Non-batched case (fast path)
    if par_array.ndim == 1 and matrix.ndim == 2:
        return jnp.dot(matrix, par_array)

    # Ensure `par_array` behaves like a batch
    if par_array.ndim == 1:
        par_array = par_array[None, :]

    # Add batch dimension to `matrix` if it’s 2D (non-batched)
    if matrix.ndim == 2:
        matrix = matrix[None, :, :]

    # Batched matrix-vector multiplication
    return jnp.einsum('...ij,...j->...i', matrix, par_array)

supersectors = {}
