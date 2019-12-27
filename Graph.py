from abc import ABC, abstractmethod
from builtins import property

from numpy import linspace
import numpy as np
from math import log
import itertools
import functools


class Domain:
    def __init__(self, values, continuous=False, integral_points=None):
        self.values = tuple(values)
        self.continuous = continuous
        if continuous:
            if integral_points is None:
                self.integral_points = linspace(values[0], values[1], 30)
            else:
                self.integral_points = integral_points

                # def __hash__(self):
                #     return hash((self.values, self.continuous))
                #
                # def __eq__(self, other):
                #     return (
                #         self.__class__ == other.__class__ and
                #         self.values == other.values and
                #         self.continuous == other.continuous
                #     )


class Potential(ABC):
    def __init__(self, symmetric=False):
        self.symmetric = symmetric
        self.alpha = 0.001

    @abstractmethod
    def get(self, parameters):
        pass

    def gradient(self, parameters, wrt):
        parameters = np.array(parameters)
        step = np.array(wrt) * self.alpha
        parameters_ = parameters + step
        return (self.get(parameters_) - self.get(parameters)) / self.alpha

    def log_gradient(self, parameters, wrt):
        parameters = np.array(parameters)
        step = np.array(wrt) * self.alpha
        parameters_ = parameters + step
        return (log(self.get(parameters_)) - log(self.get(parameters))) / self.alpha


class RV:
    id_counter = itertools.count()  # will assign unique numeric ids to instances of the class, auto-incrementing from 0

    def __init__(self, domain, value=None):
        self.domain = domain
        self.value = value
        self.id = next(self.id_counter)
        self.nb = []
        self.N = 0
        self.belief_params_ = {}  # symbolic, used by tf
        self.belief_params = {}  # np arrs
        self.sharing_count = 1  # no parameter sharing in the uncompressed/unlifted graph

    @property
    def domain_type(self):
        if self.domain.continuous:
            t = 'c-g'  # currently only support Gaussian ('g' for Gaussian)
        else:
            t = 'd-' + str(self.dstates)
        return t

    @property
    def dstates(self):
        if not self.domain.continuous:
            return len(self.domain.values)
        else:
            return None

    @property
    def values(self):  # numpy array version
        return np.array(self.domain.values)

    def __str__(self):
        return '{} rv #{}'.format(self.domain_type, self.id)

    def __lt__(self, other):
        return self.id < other.id


class F:
    id_counter = itertools.count()  # will assign unique numeric ids to instances of the class, auto-incrementing from 0

    def __init__(self, potential=None, nb=None, potential_fun=None, log_potential=None, log_potential_fun=None):
        self.potential = potential
        self.potential_fun = potential_fun  # directly callable
        self.log_potential = log_potential
        self.log_potential_fun = log_potential_fun  # directly callable; only this is used in /osi currently
        if nb is None:
            self.nb = []
        else:
            self.nb = nb
        self.id = next(self.id_counter)
        self.sharing_count = 1  # no parameter sharing in the uncompressed/unlifted graph

    @property
    def nb_domain_types(self):
        return tuple(rv.domain_type for rv in self.nb)

    @property
    def domain_type(self):
        nb_domain_types = self.nb_domain_types
        if all([t[0] == 'd' for t in nb_domain_types]):
            type = 'd'  # disc
        elif all([t[0] == 'c' for t in nb_domain_types]):
            type = 'c'  # cont
        elif self.nb:
            type = 'h'  # hybrid
        else:  # empty nb
            type = None
        return type

    def __str__(self):
        return '{}factor #{}'.format(self.domain_type, self.id)

    def __lt__(self, other):
        return self.id < other.id


# for debugging
RV.__repr__ = RV.__str__
F.__repr__ = F.__str__


class Graph:
    def __init__(self):
        self.rvs = set()
        self.factors = set()

    def init_nb(self):
        """
        Should be called after specifying self.factors, to update the neighbors of self.rvs.
        nb attributes of rvs and fs will be sorted based on ids, so they can be uniquely determined by graph topology
        :return:
        """
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors_list:
            for rv in f.nb:
                rv.nb.append(f)
        for rv in self.rvs:
            rv.N = len(rv.nb)

    @property
    @functools.lru_cache()
    def rvs_list(self):
        """
        A sorted list of self.rvs
        :return:
        """
        return list(sorted(self.rvs))

    @property
    @functools.lru_cache()
    def factors_list(self):
        """
        A sorted list of self.factors
        :return:
        """
        return list(sorted(self.factors))

    def init_rv_indices(self):
        """
        Get lists of disc/cont rvs, and build indices. These (along with the .rb attributes of rv/f) should be the only
        pieces of information used by OSI.
        :return:
        """
        Vd = [rv for rv in self.rvs_list if rv.domain_type[0] == 'd']  # list of of discrete rvs
        Vc = [rv for rv in self.rvs_list if rv.domain_type[0] == 'c']  # list of cont rvs
        Vd_idx = {n: i for (i, n) in enumerate(Vd)}
        Vc_idx = {n: i for (i, n) in enumerate(Vc)}

        self.Vd = Vd
        self.Vc = Vc
        self.Nc = len(Vc)
        self.Nd = len(Vd)
        self.Vc_idx = Vc_idx
        self.Vd_idx = Vd_idx

        # optional:
        # self.rvs_dict = {rv.id: rv for rv in self.rvs}  # in case rv.id isn't the same as its order in self.rvs

        for factor in self.factors_list:
            disc_nb_idx = ()
            cont_nb_idx = ()
            for rv in factor.nb:
                if rv.domain_type[0] == 'd':
                    disc_nb_idx += (Vd_idx[rv],)
                else:
                    assert rv.domain_type[0] == 'c'
                    cont_nb_idx += (Vc_idx[rv],)
            factor.disc_nb_idx = disc_nb_idx
            factor.cont_nb_idx = cont_nb_idx

        # for convenience
        dstates = [rv.dstates for rv in Vd]
        self.dstates = dstates

