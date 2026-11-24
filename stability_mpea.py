#!/usr/bin/env python3

from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV
from math import isfinite, log
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import warnings
from timeit import default_timer
from multiprocessing import Pool
from functools import lru_cache, partial

K=8.61733262E-5
ORDER_IM = 3
ASSETS = Path(__file__).resolve().parent / "assets"

def pretty(struct_str):
    structs = struct_str.split("_")
    if structs[1] == 'SS':
        comp = Composition(structs[0])
        if is_equimolar(comp):
            formula = ''.join(structs[0].split("1"))
        else:
            formula = structs[0]
        return formula+'_SS'+'('+structs[-1]+')'
    elif structs[-1] == 'none': 
        return structs[0]
    elif len(structs) == 3:
        return structs[0]+'('+structs[1]+'_'+structs[2]+')'
    else:
        return structs[0]+'('+structs[1]+')'

def import_mpea():
    df = pd.read_csv('expt/MPEA_dataset.csv')
    return list(set(df['FORMULA'].values))[:10]

def is_equimolar(comp):
    c = list(comp.values())
    return c.count(c[0]) == len(c)

def sane(formula):
    if not formula or len(formula) > 256:
        return False
    try:
        comp = Composition(formula, strict=True)
        if not isfinite(comp.num_atoms) or not 2 <= len(comp) <= 8:
            return False
        if len(comp.fractional_composition) != len(comp):
            return False
    except (ValueError, TypeError, OverflowError, RecursionError, ZeroDivisionError):
        return False

    allowed = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Mn', 'Mo', 'Nb', 'Ni', 'Ta', 'Ti', 'W', \
               'Zr', 'V', 'Mg', 'Re', 'Os', 'Rh', 'Ir', 'Pd', 'Pt', 'Ag', 'Au', 'Zn', 'Cd', \
               'Hg', 'Si', 'Ge', 'Ga', 'In', 'Sn', 'Sb', 'As', 'Te', 'Pb', 'Bi', 'Y', 'Sc', 'Ru']
    return all(i.symbol in allowed for i in comp.elements)

def predict(formulas, t_fac, temperature=-1, file_out=None, nproc=1):
    omegas, im, cost = init_params()
    if not type(formulas) is list:
        formulas = [formulas]
    func = partial(model,t_fac,temperature,omegas,im,cost,file_out)
    if nproc == 1:
        return [func(formula) for formula in formulas]
    with Pool(nproc) as pool:
        res = pool.map(func, formulas)
    return res

def hull_energy(diagram, candidates):
    for entry in candidates:
        if entry in diagram.stable_entries:
            return diagram.get_equilibrium_reaction_energy(entry), "stable"
    lowest = min(candidates, key=lambda entry: entry.energy_per_atom)
    return diagram.get_e_above_hull(lowest), "unstable"

def model(t_fac, temperature, omegas, im, cost, file_out, formula):
    time_0 = default_timer()
    comp_raw = Composition(formula)
    comp = comp_raw.fractional_composition
    tm = np.sum([Element(el).melting_point*comp.get_atomic_fraction(el) for el in comp.elements])
    t = temperature if temperature >= 0 else t_fac*tm
    chemsys_list=[]
    ncomp = len(comp)
    norm_dict = comp.get_el_amt_dict()
    formula_norm = ''
    for i in sorted(norm_dict.keys()):
        formula_norm += '{0}{1:.2f} '.format(i, norm_dict[i])

    # check if we have elements beyond the omegas table
    for el in comp.elements:
        if not el.symbol in omegas['elements']['BCC'].keys():
            return

    for i in range(ncomp):
        for combi in itertools.combinations(comp.elements, i + 1):
            chemsys = "-".join(sorted([x.symbol for x in combi]))
            chemsys_list.append(chemsys)
    entries=[]
    # The full system is the final combination. Keep its three competing
    # crystal structures so each hull is queried at the target composition.
    for j in chemsys_list[:-1]:
        entries.extend(compute_ss_equimolar(omegas, j, t))
    candidates = compute_ss_equimolar(omegas, chemsys_list[-1], t)
    entries.extend(candidates)
    if is_equimolar(comp):
        conf_entropy = -K*t*log(ncomp)
    else:
        candidates, conf_entropy = compute_ss(omegas, comp, t)
        entries.extend(candidates)

    energies = {e.name: e.energy_per_atom for e in candidates}
    bcc_energy = energies["SS_BCC"]
    fcc_energy = energies["SS_FCC"]
    hcp_energy = energies["SS_HCP"]
    e_above, _ = hull_energy(PhaseDiagram(entries), candidates)

    # now include the IM up to ternary
    for j in chemsys_list:
        if (len(j.split("-"))<ORDER_IM+1) & (j in im.keys()):
            for r in im[j]:
                im_energy = r['total_energy']
                im_name = Composition(r['unit_cell_formula']).reduced_formula+"_"+r["type_im"]
                entries.append(PDEntry(r['unit_cell_formula'], im_energy, name=im_name))
    pd_im=PhaseDiagram(entries)
    e_above_im, stability_im = hull_energy(pd_im, candidates)

    # dump results
    decomp=pd_im.get_decomposition(comp)
    struct = ['BCC','FCC','HCP'][np.argmin([bcc_energy,fcc_energy,hcp_energy])]
    _system = comp_raw.reduced_formula
    _cost = cost.get_cost_per_mol(comp)
    _delta_bcc = bcc_energy-fcc_energy
    _delta_hcp = hcp_energy-fcc_energy
    _decomp = str([x.name for x in decomp]).replace(' ','')
    _decomp = _decomp.strip("[]").replace("'","").split(",")
    _decomp_pretty = [pretty(i) for i in _decomp]
    hmix = enthalpy_mixing(omegas, comp, struct)
    out = {'system':_system, 'formula_norm': formula_norm, 'e_above':e_above, 'e_above_im':e_above_im, 'hmix': hmix, 'ts_conf': conf_entropy, 'stability':stability_im, \
          'phase': struct, 'cost':_cost, 'delta_bcc':_delta_bcc, 'delta_hcp':_delta_hcp, 'decomp':_decomp_pretty, 'tm':tm, 't':t}
    msg = "%s %6.3f %6.3f %6.3f %s %s %6.2f %6.3f %6.3f %s %6.0f %6.0f" \
          %(_system, e_above, e_above_im, hmix, stability_im, struct, _cost, _delta_bcc, _delta_hcp, _decomp, tm, t)
    print(msg, file=open(file_out, "a"), flush=True) if not file_out == None else print(msg, flush=True)
    return out, default_timer() - time_0, len(entries)

def compute_ss(omegas, comp, t):
    entries = []
    bcc = 0.0; fcc = 0.0; hcp = 0.0
    for i in itertools.combinations(comp.elements, 2):
        chemsys = '-'.join(sorted([el.symbol for el in i]))
        c = [comp.get_atomic_fraction(el) for el in i]
        cicj = np.prod(c)
        bcc += omegas['omegas']['BCC'][chemsys]*cicj
        fcc += omegas['omegas']['FCC'][chemsys]*cicj
        hcp += omegas['omegas']['HCP'][chemsys]*cicj
    element_ref_fcc = 0.0; element_ref_bcc = 0.0; element_ref_hcp = 0.0
    conf_entropy = 0.0
    for el in comp.elements:
        chemsys = el.symbol
        ci = comp.get_atomic_fraction(el)
        element_ref_fcc += ci*(omegas['elements']['FCC'][chemsys])
        element_ref_bcc += ci*(omegas['elements']['BCC'][chemsys])
        element_ref_hcp += ci*(omegas['elements']['HCP'][chemsys])
        conf_entropy += ci*log(ci)
    conf_entropy *= K*t
    entries.append(PDEntry(comp, (bcc+element_ref_bcc+conf_entropy),name="SS_BCC"))
    entries.append(PDEntry(comp, (fcc+element_ref_fcc+conf_entropy),name="SS_FCC"))
    entries.append(PDEntry(comp, (hcp+element_ref_hcp+conf_entropy),name="SS_HCP"))
    return entries, conf_entropy

def compute_ss_equimolar(omegas, chemsys, t):
    e=chemsys.split("-")
    entries=[]
    n=len(e)
    if n==1:
        entries.append(PDEntry(e[0]+"1",omegas['elements']['FCC'][e[0]],name=e[0]+"_FCC"))
        entries.append(PDEntry(e[0]+"1",omegas['elements']['BCC'][e[0]],name=e[0]+"_BCC"))
        entries.append(PDEntry(e[0]+"1",omegas['elements']['HCP'][e[0]],name=e[0]+"_HCP"))
    else:
        bcc=0.0; fcc=0.0; hcp=0.0
        for i in itertools.combinations(e, 2):
            bcc+=omegas['omegas']['BCC']['-'.join(i)]*(1.0/n)**2
            fcc+=omegas['omegas']['FCC']['-'.join(i)]*(1.0/n)**2
            hcp+=omegas['omegas']['HCP']['-'.join(i)]*(1.0/n)**2
        element_ref_fcc=0.0; element_ref_bcc=0.0; element_ref_hcp=0.0
        for i in e:
            element_ref_fcc+=(1.0/n)*(omegas['elements']['FCC'][i])
            element_ref_bcc+=(1.0/n)*(omegas['elements']['BCC'][i])
            element_ref_hcp+=(1.0/n)*(omegas['elements']['HCP'][i])
        ideal_entropy = -K*t*log(n)
        entries.append(PDEntry('1'.join(e)+"1",n*(bcc+element_ref_bcc+ideal_entropy),name="SS_BCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(fcc+element_ref_fcc+ideal_entropy),name="SS_FCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(hcp+element_ref_hcp+ideal_entropy),name="SS_HCP"))
    return entries

def enthalpy_mixing(omegas, comp, struct):
    hmix = 0
    for i in itertools.combinations(comp.elements, 2):
        chemsys = '-'.join(sorted([el.symbol for el in i]))
        c = [comp.get_atomic_fraction(el) for el in i]
        cicj = np.prod(c)
        hmix += cicj*omegas['omegas'][struct][chemsys]
    return hmix

@lru_cache(maxsize=1)
def init_params():
    with open(ASSETS / 'omegas.json') as f:
        omegas = json.load(f)
    with open(ASSETS / 'im_aflow_icsd.json') as f:
        im_icsd=json.load(f)
    with open(ASSETS / 'im_aflow_lib.json') as f:
        im_lib = json.load(f)
    im = {}
    keys = im_icsd.keys() | im_lib.keys()
    for k in keys:
        if im_icsd.get(k,{}) and im_lib.get(k,{}):
            im[k] = im_icsd.get(k, {}) +  im_lib.get(k,{})
        elif im_icsd.get(k,{}):
            im[k] = im_icsd.get(k, {})
        else:
            im[k] = im_lib.get(k, {})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cost=CostAnalyzer(CostDBCSV('costdb_elements.csv'))
    return omegas, im, cost

if __name__ == "__main__":
    t_fac = 0.9
    prefix_out = "mpea"
    file_out = prefix_out+'_{0}Tm_im{1}.csv'.format(t_fac, ORDER_IM)
    #formulas = import_mpea()
    formulas = 'CrCoMn0.01'
    res = predict(formulas, t_fac)
