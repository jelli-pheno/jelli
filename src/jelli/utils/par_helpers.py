import numpy as np
from itertools import product

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
