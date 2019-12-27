from Graph import *
import numpy as np
from itertools import product
import re


class LV:
    # logical variable
    def __init__(self, instances):
        self.instances = instances


class Atom:
    # relational atom
    def __init__(self, domain, logical_variables, name=None):
        self.domain = domain
        self.lvs = logical_variables
        self.name = name


class ParamF:
    # parametric factor
    def __init__(self, potential, nb=None, constrain=None):
        self.potential = potential
        self.constrain = constrain
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class RelationalGraph:
    def __init__(self, atoms, parametric_factors):
        self.atoms = atoms
        self.param_factors = parametric_factors

        # register atoms
        self.atoms_dict = dict()
        for atom in atoms:
            self.atoms_dict[atom.name] = atom

        self.rvs_dict = dict()
        self.grounding = None

    def atom_substitution(self, atom_expression, substitution):
        # atom_expression e.g. 'PartOf(s,l)' or ('PartOf', 's', 'l')
        # substitution e.g. {'s': 'seg_0', 'l': 'line_0'}
        if type(atom_expression) == str:
            expression_parts = re.findall(r'\$?\w+', atom_expression)
        else:
            expression_parts = atom_expression

        atom = self.atoms_dict[expression_parts[0]]

        key = [expression_parts[0]]
        for i in range(1, len(expression_parts)):
            s = expression_parts[i]
            if s[0] == '$':
                key.append(s[1:])
            else:
                key.append(substitution[s])

        key = tuple(key)

        rv = self.rvs_dict.get(key, RV(atom.domain))
        self.rvs_dict[key] = rv

        return key, rv

    def extract_lvs(self, atom_expression, lvs=None):
        if type(lvs) is not dict:
            lvs = dict()

        expression_parts = re.findall(r'\$?\w+', atom_expression)
        atom = self.atoms_dict[expression_parts[0]]

        for i in range(len(atom.lvs)):
            s = expression_parts[i + 1]
            if s[0] != '$':
                lvs[s] = atom.lvs[i].instances

        return lvs

    @staticmethod
    def lvs_iter(lvs):
        tokens = []
        table = []
        for token, instances in lvs.items():
            tokens.append(token)
            table.append(instances)
        for combination in product(*table):
            yield dict(zip(tokens, combination))

    def add_evidence(self, data):
        # data format: key=(RelationalAtom, LV1_instance, LV2_instance, ... ) value=True or 0.01 etc.
        for key, rv in self.rvs_dict.items():
            if key in data:
                rv.value = data[key]
            else:
                rv.value = None

        return self.grounding, self.rvs_dict

    def ground_graph(self):
        factors = set()

        # go through all parametric factors
        for param_f in self.param_factors:
            # collect lvs of neighboring atom
            lvs = dict()
            for atom_expression in param_f.nb:
                self.extract_lvs(atom_expression, lvs)

            atoms_tuple_expression = list()
            for atom_expression in param_f.nb:
                atoms_tuple_expression.append(re.findall(r'\$?\w+', atom_expression))

            for substitution in self.lvs_iter(lvs):
                if param_f.constrain is None or param_f.constrain(substitution):
                    nb = list()
                    for atom_tuple_expression in atoms_tuple_expression:
                        nb.append(self.atom_substitution(atom_tuple_expression, substitution)[1])
                    factors.add(F(potential=param_f.potential, nb=nb))

        grounding = Graph()
        grounding.rvs = set(self.rvs_dict.values())
        grounding.factors = factors
        grounding.init_nb()

        self.grounding = grounding

        return self.grounding, self.rvs_dict


from OrderedSet import OrderedSet
from collections import OrderedDict
class RelationalGraphSorted:
    def __init__(self):
        self.lvs = OrderedSet()
        self.atoms = OrderedSet()
        self.param_factors = OrderedSet()
        self.data = OrderedDict()  # format: key=(RelationalAtom, LV1_instance, LV2_instance, ... ) value=True or 0.01 etc.

    def init_nb(self):
        for atom in self.atoms:
            atom.nb = []
        for f in self.param_factors:
            for atom in f.nb:
                atom.nb.append(f)

    @staticmethod
    def lvs_iter(lvs):
        table = []
        for lv in lvs:
            table.append(lv.instances)
        for combination in product(*table):
            yield OrderedDict(zip(lvs, combination))

    def key_list(self):
        res = list()
        for atom in self.atoms:
            for substitution in self.lvs_iter(atom.lvs):
                key = atom.key(substitution)
                res.append(key)

        return res

    def grounded_graph(self):
        grounded_factors = OrderedSet()
        grounded_rvs_table = OrderedDict()

        # ground all relational atoms
        for atom in self.atoms:
            for substitution in self.lvs_iter(atom.lvs):
                key = atom.key(substitution)
                value = self.data[key] if key in self.data else None
                grounding = RV(atom.domain, value)
                grounded_rvs_table[key] = grounding

        # add factors
        for param_f in self.param_factors:
            # collect lvs of neighboring atom
            lvs = OrderedSet()
            for atom in param_f.nb:
                lvs.update(atom.lvs)
            lvs = tuple(lvs)

            # enumerate all groundings and create a factor for each grounding
            for substitution in self.lvs_iter(lvs):
                if param_f.constrain is None or param_f.constrain(substitution):
                    # collect neighboring rv instances
                    nb = []
                    for atom in param_f.nb:
                        nb.append(grounded_rvs_table[atom.key(substitution)])
                    grounded_factors.add(F(potential=param_f.potential, nb=nb))

        grounded_graph = Graph()
        tmp = OrderedSet()
        tmp.update(grounded_rvs_table.values())
        grounded_graph.rvs = tmp
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        # remove unconnected rvs
        keys = tuple(grounded_rvs_table.keys())
        for key in keys:
            if len(grounded_rvs_table[key].nb) == 0:
                del grounded_rvs_table[key]

        tmp = OrderedSet()
        tmp.update(grounded_rvs_table.values())
        grounded_graph.rvs = tmp

        return grounded_graph, grounded_rvs_table

