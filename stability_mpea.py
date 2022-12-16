#!/usr/bin/env python3

from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV
from math import log
import json
import itertools
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool
from functools import partial

K=8.61733262E-5
ORDER_IM = 3

def import_mpea():
    df = pd.read_csv('expt/MPEA_dataset.csv')
    return list(set(df['FORMULA'].values))[:10]

def sane(formula):
    try:
        comp = Composition(formula)
    except:
        return False

    allowed = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Mn', 'Mo', 'Nb', 'Ni', 'Ta', 'Ti', 'W', \
               'Zr', 'V', 'Mg', 'Re', 'Os', 'Rh', 'Ir', 'Pd', 'Pt', 'Ag', 'Au', 'Zn', 'Cd', \
               'Hg', 'Si', 'Ge', 'Ga', 'In', 'Sn', 'Sb', 'As', 'Te', 'Pb', 'Bi', 'Y', 'Sc', 'Ru']
    return all([i.symbol in allowed for i in comp.elements]) and (len(comp.elements) < 8)

def predict(formulas, t_fac, temperature=-1, file_out=None, nproc=1):
    omegas, im, cost = init_params()
    if not type(formulas) is list:
        formulas = [formulas]
    func = partial(model,t_fac,temperature,omegas,im,cost,file_out)
    with Pool(nproc) as pool:
        res = pool.map(func, formulas)
    return res

def model(t_fac, temperature, omegas, im, cost, file_out, formula):
    comp_raw = Composition(formula)
    comp = comp_raw.fractional_composition
    tm = np.sum([Element(el).melting_point*comp.get_atomic_fraction(el) for el in comp.elements])
    t = temperature if temperature >= 0 else t_fac*tm
    chemsys_list=[]
    ncomp = len(comp)

    # check if we have elements beyond the omegas table
    for el in comp.elements:
        if not el.symbol in omegas['elements']['BCC'].keys():
            return

    for i in range(ncomp):
        for combi in itertools.combinations(comp.elements, i + 1):
            chemsys = "-".join(sorted([x.symbol for x in combi]))
            chemsys_list.append(chemsys)
    entries=[]
    for j in chemsys_list:
        entries.extend(compute_ss_equimolar(omegas,j,t))
    entries.extend(compute_ss(omegas, comp, t))
    pd=PhaseDiagram(entries)
    stability="unstable"
    e_above=1E5
    for e in pd.stable_entries:
        if len(e.composition.elements) == ncomp:
            e_above=pd.get_equilibrium_reaction_energy(e)
            stability="stable"
    bcc_energy=1E5; fcc_energy=1E5; hcp_energy=1E5
    for e in pd.all_entries:
        if len(e.composition.elements) == ncomp:
            if e.name == "SS_BCC":
                bcc_energy=e.energy_per_atom
            if e.name == "SS_FCC":
                fcc_energy=e.energy_per_atom
            if e.name == "SS_HCP":
                hcp_energy=e.energy_per_atom
            if (stability=="unstable") & (e_above>pd.get_e_above_hull(e)):
                e_above=pd.get_e_above_hull(e)
    # now include the IM up to ternary
    for j in chemsys_list:
        if (len(j.split("-"))<ORDER_IM+1) & (j in im.keys()):
            for r in im[j]:
                im_energy = r['total_energy']
                im_name = Composition(r['unit_cell_formula']).reduced_formula+"_"+r["type_im"]
                entries.append(PDEntry(r['unit_cell_formula'], im_energy, name=im_name))
    pd_im=PhaseDiagram(entries)
    e_above_im=1E5
    stability_im="unstable"
    is_ss = (e.name.split('_')[-2] == 'SS')
    for e in pd_im.stable_entries:
        if (len(e.composition.elements) == ncomp) & is_ss:
            e_above_im=pd_im.get_equilibrium_reaction_energy(e)
            stability_im="stable"
    for e in pd_im.all_entries:
        if (len(e.composition.elements) == ncomp) & (stability_im == "unstable") & is_ss & (e_above_im>pd_im.get_e_above_hull(e)):
            e_above_im=pd_im.get_e_above_hull(e)

    decomp=pd_im.get_decomposition(comp)
    struct = ['BCC','FCC','HCP'][np.argmin([bcc_energy,fcc_energy,hcp_energy])]
    _system = comp_raw.reduced_formula
    _cost = cost.get_cost_per_mol(comp)
    _delta_bcc = bcc_energy-fcc_energy
    _delta_hcp = hcp_energy-fcc_energy
    _decomp = str([x.name for x in decomp]).replace(' ','')
    hmix = enthalpy_mixing(omegas, comp, struct)
    out = {'system':_system, 'e_above':e_above, 'e_above_im':e_above_im, 'hmix': hmix, 'stability':stability_im, 'phase': struct, 'cost':_cost, 'delta_bcc':_delta_bcc, 'delta_hcp':_delta_hcp, 'decomp':_decomp, 'tm':tm, 't':t}
    msg = "%s %6.3f %6.3f %6.3f %s %s %6.2f %6.3f %6.3f %s %6.0f %6.0f" %(_system, e_above, e_above_im, hmix, stability_im, struct, _cost, _delta_bcc, _delta_hcp, _decomp, tm, t)
    print(msg, file=open(file_out, "a"), flush=True) if not file_out == None else print(msg, flush=True)
    return out

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
    return entries

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
        entries.append(PDEntry('1'.join(e)+"1",n*(bcc+element_ref_bcc+ideal_entropy),name='1'.join(e)+"1"+"_SS_BCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(fcc+element_ref_fcc+ideal_entropy),name='1'.join(e)+"1"+"_SS_FCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(hcp+element_ref_hcp+ideal_entropy),name='1'.join(e)+"1"+"_SS_HCP"))
    return entries

def enthalpy_mixing(omegas, comp, struct):
    hmix = 0
    for i in itertools.combinations(comp.elements, 2):
        chemsys = '-'.join(sorted([el.symbol for el in i]))
        c = [comp.get_atomic_fraction(el) for el in i]
        cicj = np.prod(c)
        hmix += omegas['omegas'][struct][chemsys]
    return hmix*cicj

def init_params():
    with open('assets/omegas.json') as f:
        omegas = json.load(f)
    with open('assets/im_aflow_icsd.json') as f:
        im_icsd=json.load(f)
    with open('assets/im_aflow_lib.json') as f:
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
    formulas = 'Co0.125Cr0.125Mo0.1875Ni0.5625'
    res = predict(formulas, t_fac)
