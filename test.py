from random import randint, seed

seed(0)

SEGMENTS = 2
MOD = 1000
WIDTH = 10


class Point:
    def __init__(self, val=None):
        if val is not None:
            self.val = tuple(val)
        else:
            self.val = tuple(randint(0, MOD - 1) for _ in range(SEGMENTS))

    def __hash__(self) -> int:
        return hash(self.val)

    def __getitem__(self, s: int) -> int:
        return self.val[s]

    def __lt__(self, other: "Point") -> bool:
        return self.val < other.val

    def __gt__(self, other: "Point") -> bool:
        return self.val > other.val

    def __ge__(self, other: "Point") -> bool:
        return self.val >= other.val

    def __add__(self, other: "Point") -> "Point":
        return Point(tuple((self[s] + other[s]) % MOD for s in range(SEGMENTS)))

    def __sub__(self, other: "Point") -> "Point":
        return Point(tuple((self[s] - other[s]) % MOD for s in range(SEGMENTS)))

    def __neg__(self) -> "Point":
        return Point(tuple(MOD - self[s] for s in range(SEGMENTS)))

    def __eq__(self, other: "Point") -> bool:
        return all((self - other)[s] < WIDTH for s in range(SEGMENTS))

    def __repr__(self) -> str:
        return f"P{self.val}"


def brute_force(s1: list[Point], s2: list[Point], target: Point):
    res = []
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] + s2[j] == target:
                res.append((s1[i] + s2[j], s1[i], s2[j]))
    return res, len(s1) * len(s2)


def two_pointer(s1: list[Point], s2: list[Point], target: Point):
    n1, n2 = len(s1), len(s2)
    W = Point([WIDTH] * SEGMENTS)

    # ss2 = [None] * n2
    # for j in range(n2):
    #     ss2[j] = target - s2[j]
    # ss2.sort()
    # print(ss2)

    s2.reverse()
    cut = 0
    for cut in range(n2):
        if s2[cut] <= target:
            break
    ss2 = [None] * n2
    for j in range(cut):
        ss2[n2 - cut + j] = target - s2[j]
    for j in range(cut, n2):
        ss2[j - cut] = target - s2[j]
    s2 = ss2

    s1.append(-Point((1,) * SEGMENTS))

    # target <= s1 + s2 < target + W
    # target - s2 <= s1 < target + W - s2

    c = 0
    res = []
    i = 0
    for j in range(n2):
        while s1[i] < s2[j]:
            c += 1
            i += 1
        k = i
        if s1[k] == s2[j]:
            res.append((s1[k] + (target - s2[j]), s1[k], (target - s2[j])))
        # while s1[k] < s2[j] + W:
        #     c += 1
        #     if s1[k] == s2[j]:
        #         res.append((s1[k] + (target - s2[j]), s1[k], (target - s2[j])))
        #     k += 1
    return res, c


T = 1000
TARGETS = [Point() for _ in range(T)]

N1, N2 = 50, 50
S1, S2 = [Point() for _ in range(N1)], [Point() for _ in range(N2)]
S1.sort()
S2.sort()

fail = 0
sol = 0
for t in TARGETS:
    # print(f"Target: {t}")
    r1, c1 = brute_force(S1, S2, t)
    r1.sort(key=lambda x: (x[1], x[2]))

    r2, c2 = two_pointer(S1, S2.copy(), t)
    r2.sort(key=lambda x: (x[1], x[2]))
    # print(f"Brute Force: {len(r1)} results, {c1} comparisons")
    # print(f"Two Pointer: {len(r2)} results, {c2} comparisons")

    # print(r1)
    # print(r2)
    # assert len(r1) - len(r2) < 1 and not (set(r2) - set(r1)), "no false positives"
    if len(r1) > 0:
        sol += 1
        if len(r2) == 0:
            print(f"Fail {t}")
            fail += 1
            print(f"Brute Force: {len(r1)} results, {c1} comparisons")
            print(f"Two Pointer: {len(r2)} results, {c2} comparisons")

print(f"Fail {fail/sol}%")
