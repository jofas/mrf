from mrf import MondrianRandomForest, MondrianTree
from mrf import bounding_box, split

import numpy as np

from infinity import inf

X = np.array([[0.0, 1.0, 2.0],
              [1.0, 2.0, 0.0],
              [2.0, 0.0, 1.0]])

y = np.array([0, 1, 0])

def test_bounding_box():
    min, max = bounding_box(X)
    assert all([x == 0.0 for x in min])
    assert all([x == 2.0 for x in max])

def test_split():
    X_l, y_l, X_u, y_u = split(X, y, 1, 1.0)
    assert y_u[0] == 1

#def test_mondrian_tree_fit():
#    t = MondrianTree(inf)
#    t.fit(X, y, *bounding_box(X))
#    assert True == False

