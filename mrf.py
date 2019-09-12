import random
import math
import numpy as np

from infinity import inf
from scipy.stats import truncexpon

# TODO
"""
class MondrianRandomForest:
    def __init__( self, n_estimators = 10, budget=inf
                , discount = 100 ):

        self.estimators = [MondrianTree(budget, discount)
            for _ in range(n_estimators)]

    def fit(self, X, y):
        for e in self.estimators:
            e.fit(X, y)

    def update(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
"""

class MondrianTree:
    def __init__(self, budget, discount):
        self.budget   = budget
        self.discount = discount
        self.labels   = None

        self.Nil  = Nil()
        self.root = Node(self)

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.Nil.set_posterior(self.labels)

        self.root.fit(X, y)
        self.root.compute_posterior_distribution()

    def update(self, X, y):
        for x_, y_ in zip(X, y):
            self.root.update(x_, y_)
        self.root.compute_posterior_distribution()

class Node:
    def __init__(self, tree, parent = None):
        self.tree   = tree
        if parent == None:
            self.parent = tree.Nil
        else:
            self.parent = parent

        self.time  = 0
        self.dim   = 0
        self.split = 0

        self.lower = []
        self.upper = []

        self.left  = self.tree.Nil
        self.right = self.tree.Nil

        self.customers = None
        self.tables    = None

        self.posterior = {}

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts = True)

        self.lower, self.upper, deltas = \
            bounding_box(X, deltas = True)

        exp_rate = sum(self.upper - self.lower)
        if exp_rate == 0: E = inf
        else: E = random.expovariate(exp_rate)

        if self.parent.time + E >= self.tree.budget \
                or len(labels) == 1:
            self.time = self.tree.budget
            self.fit_posterior_counts(labels, counts)
            return

        self.to_inner(E, deltas)

        Xl, yl, Xu, yu = self.split_data(X, y)

        self.left  = Node(self.tree, self)
        self.right = Node(self.tree, self)

        self.left.fit(Xl, yl)
        self.right.fit(Xu, yu)

    def update(self, x, y):
        zero_vec = np.zeros(x.shape)

        # el = eu = zero_vec if x in bounding box
        el = np.maximum(self.lower - x, zero_vec)
        eu = np.maximum(x - self.upper, zero_vec)

        exp_rate = sum(el + eu)
        if exp_rate == 0: E = inf
        else: E = random.expovariate(exp_rate)

        if self.parent.time + E >= self.time \
                or (sum(self.tables.values()) == 1
                    and self.tables[y] == 1):

            # extend
            self.lower = np.minimum(self.lower, x)
            self.upper = np.maximum(self.upper, x)

            if self.time != self.tree.budget:
                if x[self.dim] <= self.split:
                    self.left.update(x, y)
                else:
                    self.right.update(x, y)
            else:
                self.update_posterior_counts(y)
        else:

            # insert parent node
            dim = random.choice(population(el + eu))
            if x[dim] > self.upper[dim]:
                split = \
                    random.uniform(self.upper[dim], x[dim])
            else:
                split = \
                    random.uniform(x[dim], self.lower[dim])

            new_parent = Node(self.tree, self.parent)
            new_parent.dim = dim
            new_parent.split = split
            new_parent.time = self.parent.time + E
            new_parent.lower = np.minimum(self.lower, x)
            new_parent.upper = np.maximum(self.upper, x)

            new_child = Node(self.tree, new_parent)

            if self == self.parent.left:
                self.parent.left = new_parent
            else:
                self.parent.right = new_parent

            if x[dim] <= split:
                new_parent.left = new_child
                new_parent.right = self
            else:
                new_parent.left = self
                new_parent.right = new_child

            self.parent = new_parent
            new_child.fit(x.reshape(1,-1), y)

    def compute_posterior_distribution(self):
        d = math.exp(-self.tree.discount
            * (self.time - self.parent.time))

        customer_sum = sum(self.customers.values())
        table_sum = sum(self.tables.values())

        if customer_sum == 0:
            self.posterior = self.parent.posterior
        else:
            for label in self.customers:
                self.posterior[label] = \
                    1 / customer_sum * (
                          self.customers[label]
                        - d * self.tables[label]
                        + d * table_sum
                        * self.parent.posterior[label]
                    )

        self.left.compute_posterior_distribution()
        self.right.compute_posterior_distribution()

    def fit_posterior_counts(
        self, labels = None, counts = None
    ):
        if self.time == self.tree.budget:
            self.customers = \
                self.compute_customers_leaf(labels, counts)
        else:
            self.customers = self.compute_customers_inner()

        self.tables = { l : min(c, 1)
            for l, c in self.customers.items() }

        self.parent.fit_posterior_counts()

    def update_posterior_counts(self, y = None):
        if self.time == self.tree.budget:
            self.customers[y] += 1
        else:
            self.customers[y] = self.left.tables[y] \
                              + self.right.tables[y]

        if self.tables[y] == 1: return

        self.tables[y] = min(self.customers[y], 1)

        self.parent.update_posterior_counts()

    def compute_customers_inner(self):
        if self.right.tables == None:
            return self.left.tables
        else:
            return { l : self.left.tables[l]
                + self.right.tables[l]
                    for l in self.tree.labels }

    def compute_customers_leaf(self, labels, counts):
        d = {}
        for l in self.tree.labels:
            idx, = np.where(labels == l)
            if len(idx) == 1:
                d[l] = counts[idx[0]]
            else:
                d[l] = 0
        return d

    def to_inner(self, E, deltas):
        self.time  = self.parent.time + E
        self.dim   = random.choice(population(deltas))
        self.split = random.uniform(
            self.lower[self.dim], self.upper[self.dim]
        )

    def split_data(self, X, y):
        idxs_l, idxs_u = [], []
        for i, x in zip(range(len(X)), X):
            idxs_l.append(i) if x[self.dim] <= self.split \
                else idxs_u.append(i)
        return X[idxs_l], y[idxs_l], X[idxs_u], y[idxs_u]

class Nil:
    def __init__(self):
        self.time = 0
        self.posterior = None

    def set_posterior(self, labels):
        self.posterior = {l : 1 / len(labels)
            for l in labels}

    def compute_posterior_distribution(self): return

    def fit_posterior_counts(self): return

def bounding_box(X, axis=0, deltas = False):
    min, max = X.min(axis = axis), X.max(axis = axis)
    if deltas: return min, max, max - min
    else: return min, max

def population(deltas):
    delta_sum = sum(deltas)

    pop = []
    for i, d in enumerate(deltas):
        norm = d / delta_sum
        pop += [i for _ in range(int(norm * 10 ** 2))]
    return pop

"""{{{
class MondrianTree:
    def __init__( self, budget, discount, parent = Nil
                , left = Nil, right = Nil, time = 0
                , dim = 0, split = 0, lower = None
                , upper = None ):

        self.budget   = budget
        self.discount = discount
        self.parent   = parent

        self.left     = left
        self.right    = right

        self.time     = time
        self.dim      = dim
        self.split    = split

        self.lower    = lower
        self.upper    = upper

        # Chinese restaurant
        self.customers = {}
        self.tables    = {}

        self.posterior = {}

    def fit(self, X, y):
        self.lower, self.upper = bounding_box(X)

        exp_rate = sum(self.upper - self.lower)
        E = random.expovariate(exp_rate)

        labels, counts = np.unique(y)

        if self.parent.time + E >= self.budget \
                or len(labels) < 2:

            # self = leaf
            self.time = self.budget
            self.init_posterior_count(labels, counts)

        else:
            # self = inner node
            self.time = self.parent.time + E

            # TODO: choose dimension proportional to
            # self.upper[dim] - self.lower[dim]
            self.dim = random.randrange(0, X.shape[1])

            self.split = random.uniform(
                self.lower[self.dim],
                self.upper[self.dim]
            )

            Xl, yl, Xu, yu = split( X, y, self.dim
                                  , self.split )

            self.left  = MondrianTree(
                self.budget, self.discount, self
            )
            self.right = MondrianTree(
                self.budget, self.discount, self
            )

            self.left.fit(Xl, yl)
            self.right.fit(Xu, yu)

    def update(self, X, y):
        for x_, y_ in zip(X, y):
            zero_vec = np.zeros(x_.shape)

            el = np.maximum(self.lower - x_, zero_vec)
            eu = np.maximum(x_ - self.upper, zero_vec)

            # TODO: incorporate pause (el = eu = 0) and
            #       node only contains one label = y

            exp_rate = sum(el + eu)
            E = random.expovariate(exp_rate)

            new_lower = np.minimum(self.lower, x_)
            new_upper = np.maximum(self.upper, x_)

            if self.parent.time + E < self.time:

                # TODO: choose dimension proportional to
                # el[dim] + eu[dim]
                pdim = random.randrange(0, X.shape[1])

                p = MondrianTree(
                    self.budget,
                    self.discount,
                    parent = self.parent,
                    time   = self.parent.time + E,
                    dim    = pdim,
                    lower  = new_lower,
                    upper  = new_upper
                )

                self.parent = p

                if x_[pdim] > self.upper[pdim]:
                    p.right = MondrianTree(
                        self.budget, self.discount, p
                    )
                    p.left  = self
                    p.split = random.uniform(
                        self.upper[pdim], x_[pdim]
                    )
                    p.right.update(x_, y_)
                else:
                    p.right = self
                    p.left  = MondrianTree(
                        self.budget, self.discount, p
                    )
                    p.split = random.uniform(
                        x_[pdim], self.lower[pdim]
                    )
                    p.left.update(x_, y_)
            else:
                self.lower = new_lower
                self.upper = new_upper

                if self.left == Nil:
                    self.update_posterior_count_leaf(y_)
                    return

                if x_[dim] <= self.split:
                    self.left.update(x_, y_)
                else:
                    self.right.update(x_, y_)

    def predict(self, X):
        for x in X:
            p_not_sep = 1
            s = { k : 0 for k in self.customers }

            return self._predict(x, s, p_not_sep)

    def _predict(self, x, s, p_not_sep):

            delta = self.time - self.parent.time

            zero_vec = np.zeros(x_.shape)

            el = np.maximum(self.lower - x, zero_vec)
            eu = np.maximum(x - self.upper, zero_vec)

            eta =  sum(el + eu)
            p_sep = 1 - math.exp(-delta * eta)

            if p_sep > 0:

                exp_d = truncexpon.mean(delta, scale=eta)

                customers = tables = {label : min(count, 1) for np.minimum(
                    self.customers.,
                    np.ones(self.customers.shape)
                )



            if self.left == Nil:
                # self = leaf
                for label in s:
                    s[label] += p_not_sep * (1 - p_sep) \
                              * self.posterior[k]
                return s
            else:
                # self = inner node
                p_not_sep *= 1 - p_sep
                if x[self.dim] <= self.split:
                    return self.left._predict(
                        x, s, p_not_sep
                    )
                else:
                    return self.right._predict(
                        x, s, p_not_sep
                    )

    def compute_posterior_distribution(self):
        if self.parent == Nil:
            self.parent.compute_posterior_distribution()

        d = math.exp(
            -self.discount * (self.time - self.parent.time)
        )

        customer_sum = sum(self.customers.values())
        table_sum = sum(self.tables.values())

        for label in self.customers:
            self.posterior[label] = \
                1 / customer_sum * (
                      self.customers[label]
                    - d * self.tables[label]
                    + d * table_sum
                    * self.parent.posterior[label]
                )

        self.left.compute_posterior_distribution()
        self.right.compute_posterior_distribution()

    def init_posterior_count_leaf(self, labels, counts):
        self.customers = dict(zip(labels, counts))
        self.tables    = dict(zip(labels, np.minimum(
            counts, np.ones(labels.shape)
        )))

        self.parent.init_posterior_count_inner(self)

    def init_posterior_count_inner(self, child):
        for label, count in child.customers.items():
            if label in self.customers:
                self.customers[label]+=child.tables[label]
            else:
                self.customers[label]=child.tables[label]

        for label, table in child.tables.items():
            self.table[label] = min(table, 1)

        self.parent.init_posterior_count_inner(self)

    def update_posterior_count_leaf(self, y):
        if y in self.customers:
            self.customers += 1
        else:
            self.customers = 1

        if y in self.tables and self.tables[y] == 1:
            return
        else:
            self.parent.update_posterior_count_inner(y)

    def update_posterior_count_inner(self, y):
        if y in self.tables and self.tables[y] == 1:
            return
        else:
            if y in self.left.customers:
                self.customers[y] = self.left.customers[y]
            else:
                self.customers[y] = 0

            if y in self.right.customers:
                self.customers[y]+=self.right.customers[y]

class Nil:
    time = 0

    labels    = []
    posterior = {}

    @static_method
    def init_posterior_count_inner(self, root):
        for label in root.customers:
            if label not in Nil.labels:
                Nil.labels.append(label)

    @static_method
    def update_posterior_count_inner(self, y):
        if y not in Nil.labels:
            Nil.labels.append(y)

    @static_method
    def compute_posterior_distribution():
        Nil.posterior = { label: 1 / len(Nillabels)
            for label in Nil.labels }
}}}"""
