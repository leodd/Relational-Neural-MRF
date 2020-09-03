from Graph import *
import numpy as np
from itertools import product
import re


class LV:
    def __init__(self, instances):
        self.instances = instances


class Atom:
    def __init__(self, domain, logical_variables, name=None):
        self.domain = domain
        self.pfs = set()
        self.lvs = logical_variables
        self.name = name

    def __call__(self, *terms):
        if len(terms) != len(self.lvs):
            raise Exception('Arity does not match.')
        return self.Atom_(self, *terms)

    class Atom_:
        def __init__(self, base, *terms):
            self.base = base
            self.terms = terms
            self.sub_idx = list()
            self.lv_idx = dict()

            for idx, term in enumerate(terms):
                if isinstance(term, str) and term[0].isupper():
                    self.sub_idx.append(idx)
                    self.lv_idx[term] = idx

        def ground(self, sub):
            terms = list(self.terms)
            for idx, ins in zip(self.sub_idx, sub):
                terms[idx] = ins
            return tuple([self.base] + terms)


class ParamF:
    def __init__(self, potential, atoms, lvs, subs=None, constrain=None):
        self.potential = potential
        self.atoms = atoms
        self.lvs = lvs

        self.lv_idx = {lv: idx for idx, lv in enumerate(lvs)}

        temp = dict()
        for atom in atoms:
            for lv, idx in atom.lv_idx.items():
                temp[lv] = atom.base.lvs[idx]
        self.lvs_ = [temp[lv] for lv in lvs]

        self.atom_sub_idx = [[self.lv_idx[atom.terms[idx]] for idx in atom.sub_idx] for atom in atoms]

        if subs is None:
            subs = product(*[lv.instances for lv in self.lvs_])

        if constrain is not None:
            subs = self.sub_filter(subs, constrain)

        self.subs = np.array(list(subs))

    @staticmethod
    def sub_filter(subs, constrain):
        for sub in subs:
            if constrain(sub):
                yield sub

    def unified_subs(self, sub, sub_idx):
        res = True
        for idx, v in zip(sub_idx, sub):
            res &= self.subs[:, idx] == v
        return self.subs[res]

    def ground(self, sub):
        return [atom.ground(sub[idx]) for atom, idx in zip(self.atoms, self.atom_sub_idx)]


class RelationalGraph:
    def __init__(self, parametric_factors):
        self.pfs = parametric_factors

    @staticmethod
    def add_evidence(rvs_dict, data):
        # data format: key=(RelationalAtom, LV1_instance, LV2_instance, ... ) value=True or 0.01 etc.
        for key, rv in rvs_dict.items():
            if key in data:
                rv.value = data[key]
            else:
                rv.value = None

    def register_rvs(self, atoms, rv_keys, rvs_dict):
        res = []
        for atom, key in zip(atoms, rv_keys):
            if key in rvs_dict:
                res.append(rvs_dict[key])
            else:
                res.append(RV(atom.base.domain))
                rvs_dict[key] = res[-1]
        return res

    def ground(self):
        fs = set()
        rvs_dict = dict()

        # go through all parametric factors
        for pf in self.pfs:
            for sub in pf.subs:
                nb = self.register_rvs(pf.atoms, pf.ground(sub), rvs_dict)
                fs.add(F(potential=pf.potential, nb=nb))

        g = Graph(set(rvs_dict.values()), fs)

        return g, rvs_dict
