from mrf import MondrianRandomForest, MondrianTree, Nil
from mrf import bounding_box, population

import numpy as np
import random

from infinity import inf

random.seed(42)

X = np.array([[0.0, 1.0, 2.0],
              [1.0, 2.0, 0.0],
              [2.0, 0.0, 1.0]])

y = np.array([0, 1, 0])

T = MondrianTree(inf, 100)
T.fit(X, y)

def test_mondrian_tree_fit():
    assert T.root.right.time == inf
    assert T.root.left.right.time == inf
    assert T.root.left.left.time == inf

def test_mondrian_tree_update():
    pass

def test_compute_posterior_distribution():
    assert sum(T.Nil.posterior.values()) == 1.0
    assert sum(T.root.right.posterior.values()) == 1.0
    assert sum(T.root.left.posterior.values()) == 1.0
    assert sum(T.root.left.right.posterior.values()) == 1.0
    assert sum(T.root.left.left.posterior.values()) == 1.0

def test_posterior_counts():
    assert T.root.left.left.customers == {0: 1, 1: 0}
    assert T.root.left.right.customers == {0: 0, 1: 1}
    assert T.root.right.customers == {0: 1, 1: 0}

    assert T.root.left.customers == {0: 1, 1: 1}
    assert T.root.customers == {0: 2, 1: 1}

    assert T.root.tables == {0: 1, 1: 1}

def test_bounding_box():
    min, max = bounding_box(X)
    assert all([x == 0.0 for x in min])
    assert all([x == 2.0 for x in max])

def test_bounding_box_deltas():
    min, max, deltas = bounding_box(X, deltas = True)
    assert all([x == 2.0 for x in deltas])

def test_bounding_box_axis():
    min, max = bounding_box(X[0], axis=0)
    assert min == 0.0 and max == 2.0

def test_population():
    min, max, deltas = bounding_box(X, deltas = True)
    pop = population(deltas)
    assert all([x == 0 for x in pop[:33]])

def test_split_data():
    t = MondrianTree(inf, 100)
    t.root.dim = 1
    t.root.split = 1.0
    X_l, y_l, X_u, y_u = t.root.split_data(X, y)
    assert y_u[0] == 1

def test_compute_customers_leaf():
    t = MondrianTree(inf, 100)
    t.labels = np.array([0, 1])
    customers = t.root.compute_customers_leaf(
        np.array([1]), [3]
    )

    assert customers[0] == 0
    assert customers[1] == 3

    customers = t.root.compute_customers_leaf(
        np.array([]), []
    )

    assert customers[0] == 0
    assert customers[1] == 0

def test_Nil_set_posterior():
    labels = ["test1", "test2", 0, 5]

    nil = Nil()
    nil.set_posterior(labels)

    assert nil.posterior[0] == 1 / 4

    nil.set_posterior(np.array(labels))

    assert nil.posterior["0"] == 1 / 4
