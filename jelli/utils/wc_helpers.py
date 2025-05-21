import numpy as np
import jax.numpy as jnp

def get_wc_basis_from_wcxf(eft, basis, sector=None, split_re_im=True):
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
    basis_full = get_wc_basis_from_wcxf(eft, basis)
    return np.concatenate([
        [basis_full.index(wc) for wc in get_wc_basis_from_wcxf(eft, basis, sector)]
        for sector in sectors
    ])
