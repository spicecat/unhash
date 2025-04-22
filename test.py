from random import randint, seed

seed(0)

SEGMENTS = 2
MOD = 200
WIDTH = 10


class Point:
    def __init__(self, val=None):
        if val is not None:
            self.val = val
        else:
            self.val = tuple(randint(0, MOD - 1) for _ in range(SEGMENTS))

    def __lt__(self, other):
        return self.val < other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __add__(self, other):
        return Point(tuple((self.val[s] + other.val[s]) % MOD for s in range(SEGMENTS)))

    def __sub__(self, other):
        return Point(tuple((self.val[s] - other.val[s]) % MOD for s in range(SEGMENTS)))

    def __eq__(self, other):
        return all((self.val[s] - other.val[s]) % MOD < WIDTH for s in range(SEGMENTS))

    def __repr__(self):
        return f"P{self.val}"


def naive(S1, S2, TARGET):
    res = []
    for i in range(len(S1)):
        for j in range(len(S2)):
            if S1[i] + S2[j] == TARGET:
                res.append((S1[i] + S2[j]))
    return len(res), res, len(S1) * len(S2)


def two_pointer(S1, S2, TARGET):
    S1P = [[] for _ in range(len(S1))]

    i = 0
    c = 0
    res = []
    for j in range(N2):
        s = i
        while S1[i] + S2[j] >= TARGET:
            i = (i - 1) % N1
            c += 1
            if i == s:
                break
        s = i
        while S1[i] + S2[j] < TARGET:
            if S1[i] + S2[j] == TARGET:
                res.append((S1[i] + S2[j]))
            i = (i - 1) % N1
            c += 1
            if i == s:
                break
        i = s
    return len(res), res, c


TARGET = Point()
print(TARGET)

N1 = 50
S1 = [Point() for _ in range(N1)]
S1.sort()
N2 = 50
S2 = [Point() for _ in range(N2)]
S2.sort()

print(naive(S1, S2, TARGET))
print(two_pointer(S1, S2, TARGET))
