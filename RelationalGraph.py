from Graph import *
import numpy as np
from itertools import product
from collections import defaultdict
import re


class LV:
    def __init__(self, instances):
        self.instances = instances


class Atom:
    def __init__(self, domain, logical_variables, name=None):
        self.domain = domain
        self.pfs = list()
        self.lvs = logical_variables
        self.name = name

    def __call__(self, *terms, subs=None):
        if len(terms) != len(self.lvs):
            raise Exception('Arity does not match.')
        return self.Atom_(self, subs, *terms)

    class Atom_:
        def __init__(self, base, subs, *terms):
            self.base = base
            self.subs = subs
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

        if isinstance(subs, np.ndarray):
            self.subs = subs
        else:
            self.subs = np.array(list(subs))

    @staticmethod
    def sub_filter(subs, constrain):
        for sub in subs:
            if constrain(sub):
                yield sub

    def unified_subs(self, subs, sub_idx, return_mask=False):
        res = np.apply_along_axis(lambda r: tuple in subs, 1, self.subs[:, sub_idx])
        return res if return_mask else self.subs[res]

    def ground(self, sub):
        return [atom.ground(sub[idx]) for atom, idx in zip(self.atoms, self.atom_sub_idx)]


class RelationalGraph:
    def __init__(self, parametric_factors):
        self.pfs = parametric_factors
        self.init_atom_pfs(parametric_factors)

    @staticmethod
    def init_atom_pfs(pfs):
        for pf in pfs:
            for idx, atom in enumerate(pf.atoms):
                atom.base.pfs.append((pf, idx))

    @staticmethod
    def add_evidence(rvs_dict, data):
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

    def ground(self, data=None):
        fs = set()
        rvs_dict = dict()

        # go through all parametric factors
        for pf in self.pfs:
            for sub in pf.subs:
                nb = self.register_rvs(pf.atoms, pf.ground(sub), rvs_dict)
                fs.add(F(potential=pf.potential, nb=nb))

        g = Graph(set(rvs_dict.values()), fs)

        if data is not None:
            self.add_evidence(rvs_dict, data)

        return g, rvs_dict

    def partial_ground(self, queries, data, depth=2):
        fs = set()
        rvs_dict = dict()

        atom_subs = defaultdict(set)
        pf_subs_mask = defaultdict(bool)

        new_atom_subs = defaultdict(set)
        atom_obs = defaultdict(set)


        for key in queries:
            new_atom_subs[key[0]].add(key[1:])

        if data is not None:
            for key in data:
                atom_obs[key[0]].add(key[1:])

        for _ in range(depth):
            pf_temp_mask = defaultdict(bool)

            for atom, subs in new_atom_subs.items():
                for pf, idx in atom.pfs:
                    mask = pf.unified_subs(subs, pf.atom_sub_idx[idx], return_mask=True)
                    pf_temp_mask[pf] |= mask
                    pf_subs_mask[pf] |= mask

            new_atom_subs = defaultdict(set)

            for pf, mask in pf_temp_mask.items():
                for idx, atom in enumerate(pf.atoms):
                    subs = pf.subs[np.ix_(mask, pf.atom_sub_idx[idx])]
                    new_atom_subs[atom.base].update(map(tuple, subs))

            for atom in new_atom_subs:
                new_atom_subs[atom] -= atom_subs[atom] | atom_obs[atom]
                atom_subs[atom].update(new_atom_subs[atom])

        for pf in pf_subs_mask:
            for sub in pf.subs[pf_subs_mask[pf]]:
                nb = self.register_rvs(pf.atoms, pf.ground(sub), rvs_dict)
                fs.add(F(potential=pf.potential, nb=nb))

        g = Graph(set(rvs_dict.values()), fs)

        if data is not None:
            self.add_evidence(rvs_dict, data)

        return g, rvs_dict
