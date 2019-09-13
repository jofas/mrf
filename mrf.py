import random
import math
import numpy as np

from scipy.stats import truncexpon

from sys import float_info

class MondrianRandomForest:
    def __init__(
        self, n_estimators = 10, budget=np.inf,
        discount = 100
    ):
        self.estimators = [MondrianTree(budget, discount)
            for _ in range(n_estimators)]

    def fit(self, X, y):
        for e in self.estimators:
            e.fit(X, y, called_from_forest = True)

    def update(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

class DummyForest:
    def __init__(self):
        self.labels = None

class MondrianTree:
    def __init__(
        self, budget, discount, forest = DummyForest()
    ):
        self.budget   = budget
        self.discount = discount
        self.forest   = forest

        self.Nil  = Nil()
        self.root = Node(self)

    def fit(self, X, y, called_from_forest = False):
        if not called_from_forest:
            self.forest.labels = np.unique(y)
        self.Nil.set_posterior(self.forest.labels)

        self.root.fit(X, y)
        self.root.compute_posterior_distribution()

    def update(self, X, y):
        for x_, y_ in zip(X, y):
            self.root.update(x_, y_)
        self.root.compute_posterior_distribution()

    def predict(self, X):
        return [self.root.predict(x) for x in X]

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
        if exp_rate == 0: E = np.inf
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
        if exp_rate == 0: E = np.inf
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

    def predict(self, x, p_not_sep = None, probas = None):
        if p_not_sep == None:
            p_not_sep = 1.0
            probas = {l: 0.0 for l in self.customers}

        time_delta = self.time - self.parent.time

        if time_delta == np.inf:
            time_delta = float_info.max

        zero_vec = np.zeros(x.shape)
        el = np.maximum(self.lower - x, zero_vec)
        eu = np.maximum(x - self.upper, zero_vec)
        eta = sum(el + eu)

        p_sep = 1.0 - math.exp(-time_delta * eta)

        if p_sep > 0.0:
            exp_d = truncexpon.mean(time_delta, scale=eta)
            customers = tables = self.tables

            customers_sum = sum(customers.values())
            tables_sum    = sum(tables.values())

            for label in customers:
                posterior_label = \
                    1 / customers_sum * (
                          customers[label]
                        - exp_d * tables[label]
                        + exp_d * tables_sum
                        * self.parent.posterior[label]
                    )
                probas[label] += p_not_sep * p_sep \
                               * posterior_label

        if self.time == self.tree.budget:
            for label in self.customers:
                probas[label] += p_not_sep * (1 - p_sep) \
                               * self.posterior[label]
            return probas
        else:
            p_not_sep = p_not_sep * (1 - p_sep)
            if x[self.dim] <= self.split:
                return self.left.predict(
                    x, p_not_sep, probas
                )
            else:
                return self.right.predict(
                    x, p_not_sep, probas
                )

    def compute_posterior_distribution(self):
        d = math.exp(-self.tree.discount
            * (self.time - self.parent.time))

        customer_sum = sum(self.customers.values())
        table_sum    = sum(self.tables.values())

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
                    for l in self.tree.forest.labels }

    def compute_customers_leaf(self, labels, counts):
        d = {}
        for l in self.tree.forest.labels:
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
