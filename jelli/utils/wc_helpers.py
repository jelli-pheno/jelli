import numpy as np

def get_wc_basis_from_wcxf(eft, basis, sector=None):
    from wilson import wcxf
    basis_obj = wcxf.Basis[eft, basis]
    wc_list = []

    if sector and sector not in basis_obj.sectors.keys():
        raise ValueError(f"Sector {sector} not found in basis {basis} of EFT {eft}")

    for sec, s in basis_obj.sectors.items():
        if not sector or sec == sector:
            for name, d in s.items():
                if not d or 'real' not in d or not d['real']:
                    wc_list.append((name, 'R'))
                    wc_list.append((name, 'I'))
                else:
                    wc_list.append((name, 'R'))
    return sorted(wc_list)

def get_sector_indices_from_wcxf(eft, basis, sectors):
    basis_full = get_wc_basis_from_wcxf(eft, basis)
    return np.concatenate([
        [basis_full.index(wc) for wc in get_wc_basis_from_wcxf(eft, basis, sector)]
        for sector in sectors
    ])
