from Graph import *
from collections import Counter
from statistics import mean
from random import uniform
import numpy as np


class SuperRV:
    def __init__(self, rvs, domain=None):
        self.rvs = rvs
        self.domain = next(iter(rvs)).domain if domain is None else domain
        self.nb = None
        self.N = 0
        for rv in rvs:
            rv.cluster = self

    def __lt__(self, other):
        return hash(self) < hash(other)

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def update_nb(self):
        rv = next(iter(self.rvs))
        self.nb = tuple(map(self.get_cluster, rv.nb))
        self.N = rv.N

    def split_by_structure(self):
        clusters = dict()
        for rv in self.rvs:
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))
            if signature in clusters:
                clusters[signature].add(rv)
            else:
                clusters[signature] = {rv}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.rvs = clusters[next(i)]
        res.add(self)

        for _ in range(1, len(clusters)):
            res.add(SuperRV(clusters[next(i)], self.domain))

        return res


class SuperF:
    def __init__(self, factors):
        self.factors = factors
        self.potential = next(iter(factors)).potential
        self.nb = None
        for f in factors:
            f.cluster = self

    def __lt__(self, other):
        return hash(self) < hash(other)

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def update_nb(self):
        f = next(iter(self.factors))
        self.nb = tuple(map(self.get_cluster, f.nb))

    def split_by_structure(self):
        clusters = dict()
        for f in self.factors:
            signature = tuple(sorted(map(self.get_cluster, f.nb))) \
                if f.potential.symmetric else tuple(map(self.get_cluster, f.nb))
            if signature in clusters:
                clusters[signature].add(f)
            else:
                clusters[signature] = {f}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.factors = clusters[next(i)]
        res.add(self)

        for _ in range(1, len(clusters)):
            res.add(SuperF(clusters[next(i)]))

        return res


class CompressedGraph:
    # color passing algorithm for compressing graph

    def __init__(self, graph):
        self.g = graph
        self.rvs = set()
        self.factors = set()

    def init_cluster(self):
        self.rvs.clear()
        self.factors.clear()

        # group rvs according to domain
        color_table = dict()
        for rv in self.g.rvs:
            if rv.domain in color_table:
                color_table[rv.domain].add(rv)
            else:
                color_table[rv.domain] = {rv}
        for _, cluster in color_table.items():
            self.rvs.add(SuperRV(cluster))

        # group factors according to potential
        color_table.clear()
        for f in self.g.factors:
            if f.potential in color_table:
                color_table[f.potential].add(f)
            else:
                color_table[f.potential] = {f}
        for _, cluster in color_table.items():
            self.factors.add(SuperF(cluster))

    def split_rvs(self):
        for rv in tuple(self.rvs):
            self.rvs |= rv.split_by_structure()

    def split_factors(self):
        for f in tuple(self.factors):
            self.factors |= f.split_by_structure()

    def run(self):
        self.init_cluster()

        prev_rvs_num = -1
        while prev_rvs_num != len(self.rvs):
            prev_rvs_num = len(self.rvs)
            self.split_factors()
            self.split_rvs()

        for rv in self.rvs:
            rv.update_nb()
        for f in self.factors:
            f.update_nb()
