from fractions import Fraction
import operator
from numbers import Number


class Vector():
    def __init__(self, iterable):
        self.v = list(iterable)
        assert len(self.v) > 0, "cannot create 0-th dimensional vector"

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return f"Vector({self.v})"

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]

    def __setitem__(self, i, x):
        self.v[i] = x

    def __add__(self, other):
        if type(other) is not Vector:
            return NotImplemented
        assert len(other) == len(self), 'Vectors must be of the same length'
        return Vector([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return self + -1 * other

    def __mul__(self, number):
        if not isinstance(number, Number):
            return NotImplemented
        return self.scale(number)

    def __neg__(self):
        return -1 * self

    def __eq__(self, other):
        assert type(other) is Vector
        return self.v == other.v

    def __rmul__(self, number):
        return self.__mul__(number)

    def __round__(self, r=0):
        return Vector(round(float(i), r) for i in self.v)

    def __or__(self, other):
        if type(other) is Vector:
            return self.v + other.v
        return self.v + [other]

    def __gt__(self, other):
        assert len(self) == len(
            other), "can only compare Vectors of the same size"
        return all(a > b for a, b in zip(self, other))

    @property
    def T(self):
        from .matrix import Matrix
        return Matrix([self.v])

    def copy(self):
        return Vector(self.v.copy())

    @property
    def shape(self):
        return len(self.v), 1

    def magnitude(self, norm=2):
        if norm == float('inf'):
            return max(self.v)
        return (sum(x**norm for x in self.v))**(1 / norm)

    def dot(self, v):
        assert len(self) == len(
            v), 'dot product only works between to Vectors of the same size'
        return sum(map(operator.mul, self, v))

    @staticmethod
    def cross(u, v):
        assert len(u) == len(v) == 3
        return Vector([
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
        ])

    def scale(self, s):
        return Vector(a * s for a in self.v)

    @staticmethod
    def to_unit(iterable, norm=2):
        v = Vector(iterable)
        return v.scale(1 / v.magnitude(norm))

    @staticmethod
    def to_fraction_unit(lst):
        n = sum(lst)
        return Vector([Fraction(i, n) for i in lst])

    @staticmethod
    def fill(n, x=0):
        return Vector([x] * n)

    def filter(self, entries):
        v = [v for i, v in enumerate(self.v) if i in entries]
        if not v:
            return None
        return Vector(v)

    def evaluate(self, dict):
        ans = []
        for i in self:
            if hasattr(i, 'evaluate'):
                ans.append(i.evaluate(dict))
            else:
                ans.append(i)
        return Vector(ans)
