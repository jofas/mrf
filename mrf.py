import random

from infinity import inf

class MondrianRandomForest:
    def __init__(self, n_estimators=10, budget=inf):
        self.estimators = [MondrianTree(budget)
            for _ in range(n_estimators)]

    def fit(self, X, y):
        for e in self.estimators: e.fit(X, y)

    def update(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

class MondrianTree:
    def __init__( self, budget, parent = Nil, left = Nil
                , right = Nil, time = 0, dim = 0
                , split = 0, lower = None, upper = None ):
        self.budget = budget

        self.parent = parent

        self.left   = left
        self.right  = right

        self.time  = time
        self.dim   = dim
        self.split = split

        self.lower = lower
        self.upper = upper

    def fit(self, X, y):
        self.lower, self.upper = bounding_box(X)

        # paused
        #if len(np.unique(y)) < 2:
        #    self.time = self.budget
        #    return

        exp_rate = sum(self.upper - self.lower)
        E = random.expovariate(exp_rate)

        if self.parent.time + E < self.budget:
            self.time = self.parent.time + E

            # TODO: choose dimension proportional to
            # u[dim] - l[dim]
            self.dim = random.randrange(0, X.shape[1])

            self.split = random.uniform( l[self.dim]
                                       , u[self.dim] )

            Xl, yl, Xu, yu = split( X, y, self.dim
                                  , self.split )

            self.left  = MondrianTree(self.budget, self)
            self.right = MondrianTree(self.budget, self)

            self.left.fit(Xl, yl)
            self.right.fit(Xu, yu)

        else:
            self.time = self.budget
            # TODO: build probability distro

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
                    parent = self.parent,
                    time   = self.parent.time + E,
                    dim    = pdim,
                    lower  = new_lower,
                    upper  = new_upper
                )

                self.parent = p

                if x_[pdim] > self.upper[pdim]:
                    p.right = MondrianTree(self.budget, p)
                    p.left  = self
                    p.split = random.uniform(
                        self.upper[pdim], x_[pdim]
                    )
                    p.right.update(x_, y_)
                else:
                    p.right = self
                    p.left  = MondrianTree(self.budget, p)
                    p.split = random.uniform(
                        x_[pdim], self.lower[pdim]
                    )
                    p.left.update(x_, y_)
            else:
                self.lower = new_lower
                self.upper = new_upper

                if x_[dim] <= self.split:
                    self.left.update(x_, y_)
                else:
                    self.right.update(x_, y_)

    def predict(self, X):
        # TODO
        pass

class Nil:
    time = 0

    @staticmethod
    def update(X, y): return

def bounding_box(X):
    return X.min(axis=1), X.max(axis=1)

def split(X, y, dim, split):
    idxs_l, idxs_u = [], []
    for i, x in zip(range(len(X)), X):
        idxs_l.append(i) if x[dim] <= split \
            else idxs_u.append(i)
    return X[idxs_l], y[idxs_l], X[idxs_u], y[idxs_u]
