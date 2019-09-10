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

def test_mondrian_tree_fit():
    t = MondrianTree(inf, 100)
    t.fit(X, y)
    # all the leaf nodes
    assert t.root.right.time == inf
    assert t.root.left.right.time == inf
    assert t.root.left.left.right.right.time == inf
    assert t.root.left.left.right.left.time == inf
    assert t.root.left.left.left.time == inf

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

def test_Nil_set_posterior():
    labels = ["test1", "test2", 0, 5]

    nil = Nil()
    nil.set_posterior(labels)

    assert nil.posterior[0] == 1 / 4

    nil.set_posterior(np.array(labels))

    assert nil.posterior["0"] == 1 / 4

