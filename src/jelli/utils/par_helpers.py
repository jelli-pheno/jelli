import warnings
import numpy as np
from itertools import product
from typing import List
from rgevolve.tools import get_wc_basis


def get_wc_mask(eft, basis, sector, wcs):
    '''
    Get a mask that selects the Wilson coefficients `wcs` in a sector.

    Deprecated, and to be removed: jelli does not use it any more. `rgevolve.tools.run_and_match`
    returns the rows of these coefficients directly when they are passed as `wcs_out`.

    Parameters
    ----------
    eft : str
        The effective field theory (EFT) name, e.g., `SMEFT` or `WET`.
    basis : str
        The Wilson coefficient basis name, e.g., `Warsaw` or `JMS`.
    sector : str
        The RGE sector.
    wcs : list
        Wilson coefficients of the sector, as tuples `(name, 'R')` or `(name, 'I')`.

    Returns
    -------
    np.ndarray
        A boolean array over `get_wc_basis(eft, basis, sector)` that is `True` at the coefficients in `wcs`.
    '''
    warnings.warn(
        "get_wc_mask is deprecated and will be removed from jelli. "
        "Pass the coefficients to rgevolve.tools.run_and_match as wcs_out instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    wc_basis_sector = get_wc_basis(eft, basis, sector)
    if not all([wc in wc_basis_sector for wc in wcs]):
        raise ValueError(f"Invalid coefficients in sector {sector} of basis {basis} of EFT {eft}")
    return np.array([wc in wcs for wc in wc_basis_sector])


def get_sector_indices(eft: str, basis: str, sectors: List[str]) -> np.ndarray:
    '''
    Get the positions of the Wilson coefficients of some sectors in the full basis.

    Deprecated, and to be removed: jelli does not use it any more. `rgevolve.tools.run_and_match`
    with `sector_out=None` takes the coefficients of several sectors as `wcs_in`, so that the
    columns of its matrix match a list of coefficients chosen by the caller.

    Parameters
    ----------
    eft : str
        The effective field theory (EFT) name, e.g., `SMEFT` or `WET`.
    basis : str
        The Wilson coefficient basis name, e.g., `Warsaw` or `JMS`.
    sectors : list
        The RGE sectors.

    Returns
    -------
    np.ndarray
        The positions in `get_wc_basis(eft, basis)` of the coefficients of each sector, sector by
        sector, each in the order of `get_wc_basis(eft, basis, sector)`.
    '''
    warnings.warn(
        "get_sector_indices is deprecated and will be removed from jelli. Pass the coefficients "
        "to rgevolve.tools.run_and_match as wcs_in instead, or take their positions in "
        "rgevolve.tools.get_wc_basis(eft, basis).",
        DeprecationWarning,
        stacklevel=2,
    )
    basis_full = get_wc_basis(eft, basis)
    return np.concatenate([
        [basis_full.index(wc) for wc in get_wc_basis(eft, basis, sector)]
        for sector in sectors
    ])


def get_wc_basis_from_wcxf(eft, basis, sector=None, split_re_im=True):
    '''
    Retrieve the Wilson coefficient basis from WCxf.

    Parameters
    ----------
    eft : str
        The effective field theory (EFT) name, e.g., `SMEFT` or `WET`.
    basis : str
        The Wilson coefficient basis name, e.g., `Warsaw` or `JMS`.
    sector : str, optional
        The RGE sector of interest. If `None`, all sectors are included.
    split_re_im : bool, optional
        If `True`, split complex Wilson coefficients into their real and imaginary parts.
        If `False`, keep them as complex parameters. Default is `True`.

    Returns
    -------
    list
        A sorted list of Wilson coefficient names. If `split_re_im` is `True`,
        complex coefficients are represented as tuples `(name, 'R')` and `(name, 'I')`
        for their real and imaginary parts, respectively.
    '''
    from wilson import wcxf
    basis_obj = wcxf.Basis[eft, basis]
    wc_list = []

    if sector and sector not in basis_obj.sectors.keys():
        raise ValueError(f"Sector {sector} not found in basis {basis} of EFT {eft}")

    if split_re_im:
        for sec, s in basis_obj.sectors.items():
            if not sector or sec == sector:
                for name, d in s.items():
                    if not d or 'real' not in d or not d['real']:
                        wc_list.append((name, 'R'))
                        wc_list.append((name, 'I'))
                    else:
                        wc_list.append((name, 'R'))
    else:
        for sec, s in basis_obj.sectors.items():
            if not sector or sec == sector:
                for name, d in s.items():
                    wc_list.append(name)
    return sorted(wc_list)

def get_sector_indices_from_wcxf(eft, basis, sectors):
    '''
    Get indices of Wilson coefficients from the full basis corresponding to specified sectors.

    Parameters
    ----------
    eft : str
        The effective field theory (EFT) name, e.g., `SMEFT` or `WET`.
    basis : str
        The Wilson coefficient basis name, e.g., `Warsaw` or `JMS`.
    sectors : list
        A list of sector names to extract indices for.

    Returns
    -------
    np.ndarray
        An array of indices corresponding to the Wilson coefficients in the specified sectors.
    '''
    basis_full = get_wc_basis_from_wcxf(eft, basis)
    return np.concatenate([
        [basis_full.index(wc) for wc in get_wc_basis_from_wcxf(eft, basis, sector)]
        for sector in sectors
    ])

# useful functions for dealing with parameter linear and bilinear keys
def keys_product(keys_a, keys_b):
    """
    Computes the Cartesian product of two sets of keys, producing bilinear combinations.

    Parameters
    ----------
    keys_a: list
        A list where each element is a tuple of the form (w, c).
    keys_b: list
        Another list with elements of the form (w, c).

    Returns
    -------
    list
        A list of bilinear combinations in the form (w_a, w_b, c_a + c_b).
    """
    if len(keys_a[0]) == 2:
        return [
            (w_a, w_b, c_a+c_b)
            for ((w_a, c_a), (w_b, c_b)) in product(keys_a, keys_b)
        ]
    else:
        raise ValueError("keys must be of the form (w,c)")

def keys_array(keys):
    """
    Converts a list of tuples into a numpy array.

    Parameters
    ----------
    keys : list
        A list containing tuples.

    Returns
    -------
    `np.ndarray`
        A numpy array with dtype=tuple containing the provided keys.
    """
    array = np.empty(len(keys), dtype=tuple)
    array[:] = keys
    return array

def keys_isin(keys_a, keys_b):
    """
    Checks if elements in `keys_a` exist in `keys_b`.

    Parameters
    ----------
    keys_a : list
        List of keys to check.
    keys_b : list
        List of reference keys.

    Returns
    -------
    `np.ndarray`
        Boolean numpy array indicating presence of each key in `keys_a` within `keys_b`.
    """
    set_b = set(keys_b)
    res = np.array([item in set_b for item in keys_a])
    return res if res.size > 0 else np.array([], dtype=bool)

def get_par_monomial_indices(keys_pars, keys_coeff):
    """
    Computes sorted indices mapping linear parameters
    to bilinear ones that exist in the provided coefficient list.

    Parameters
    ----------
    keys_pars : list
        List of linear parameter keys, each element is a tuple `(w, c)`,
        where `w` is the parameter name and `c` is `R` for real or `I` for imaginary.
    keys_coeff : list
        List of bilinear coefficient keys. Each element is a tuple `(w1, w2, c)`,
        where `w1` and `w2` are the parameter names and `c` is `RR`, `RI`, `IR`, or `II`,
        denoting all possible interferences.

    Returns
    -------
    `np.ndarray`
        Sorted indices of bilinear coefficients that match `keys_coeff`.
    """
    # Generate all possible bilinear combinations of keys_pars
    keys_pars_bilinears = keys_array(keys_product(
        keys_pars, keys_pars
    ))
    bilin_bools = keys_isin(keys_pars_bilinears, keys_coeff)
    # Take elements of keys_pars_bilinears that exist in keys_coeff and obtain indices that sort them
    sort_indices = np.argsort(keys_pars_bilinears[bilin_bools])
    bilin_indices = np.where(bilin_bools)[0]
    bilin_sort_indices = bilin_indices[sort_indices]
    return bilin_sort_indices

def get_active_monomial_indices(keys_pars, keys_coeff, sector_indices, evolution_matrices, par_indices):
    """
    Computes the indices of the parameter monomials that can be nonzero
    when only the parameters at `par_indices` are nonzero.

    A monomial is the product of two evolved parameters of an observable sector,
    or of an evolved parameter and the constant. An evolved parameter vanishes
    identically if its row of the evolution matrices is zero, at every scale,
    in the columns of the parameters at `par_indices`. Without RG mixing, when
    the evolution matrix only selects parameters, the evolved parameters that
    do not vanish are the selected ones at `par_indices`.

    Parameters
    ----------
    keys_pars : list
        Keys of the evolved parameters, each a tuple `(w, c)`. The first key is
        the constant `('', 'R')`.
    keys_coeff : list
        Keys of the monomials, each a tuple `(w1, w2, c)`, in the order of the
        monomials (see `get_par_monomial_indices`).
    sector_indices : np.ndarray
        Indices in the full parameter array of the columns of `evolution_matrices`.
    evolution_matrices : np.ndarray
        Evolution matrices of shape `(n_scales, len(keys_pars) - 1, len(sector_indices))`.
    par_indices : np.ndarray
        Indices in the full parameter array of the parameters that can be nonzero.

    Returns
    -------
    `np.ndarray`
        Sorted indices of the monomials that can be nonzero.
    """
    columns = np.isin(np.asarray(sector_indices), np.asarray(par_indices))
    nonzero = np.any(np.asarray(evolution_matrices)[:, :, columns] != 0, axis=(0, 2))
    active = np.concatenate([[True], nonzero])  # the constant is always active
    first, second = np.divmod(get_par_monomial_indices(keys_pars, keys_coeff), len(keys_pars))
    return np.flatnonzero(active[first] & active[second])
