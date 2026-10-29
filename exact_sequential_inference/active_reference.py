"""Independent unnormalized-risk oracle and bounded full policy enumeration.

No production posterior or Bellman routine is imported. Only strict parsing and
rational serialization are shared. The tiny enumerator constructs every policy
loss vector before minimizing; it does not prune suboptimal child policies.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
from .active import validate
from .common import qstr, require


def solve(inp):
    validate(inp)
    n, m = len(inp["prior"]), len(inp["tests"])
    p1 = [[Fraction(s) for s in row] for row in inp["p_one"]]
    costs = [Fraction(s) for s in inp["costs"]]
    terminal = [tuple(Fraction(s) for s in row) for row in inp["loss"]]
    w = [Fraction(s) for s in inp["prior"]]
    available = list(range(m))
    for event in inp["history"]:
        j, b = event["test"], event["outcome"]
        available.remove(j)
        for h in range(n):
            w[h] *= p1[j][h] if b == 1 else 1 - p1[j][h]
    mass = sum(w)
    if not mass:
        return {"status": "impossible_evidence", "evidence_probability": "0/1", "posterior": None, "action_values": [], "minimum_expected_loss": None, "selected_action": None}, []

    @lru_cache(None)
    def candidates(weights, left, depth):
        vals = [(a, sum(weights[h] * row[h] for h in range(n))) for a, row in enumerate(terminal)]
        if depth:
            for t in left:
                future = tuple(x for x in left if x != t)
                z = tuple(weights[h] * (1 - p1[t][h]) for h in range(n))
                o = tuple(weights[h] * p1[t][h] for h in range(n))
                v = sum(weights) * costs[t]
                v += min(x[1] for x in candidates(z, future, depth - 1))
                v += min(x[1] for x in candidates(o, future, depth - 1))
                vals.append((n + t, v))
        return tuple(vals)

    vals = candidates(tuple(w), tuple(available), inp["initial_horizon"] - len(inp["history"]))
    best = min(v for a, v in vals)
    opts = [a for a, v in vals if v == best]
    return {"status": "ok", "evidence_probability": qstr(mass), "posterior": [qstr(v / mass) for v in w], "action_values": [[a, qstr(v / mass)] for a, v in vals], "minimum_expected_loss": qstr(best / mass), "selected_action": min(opts)}, opts


def enumerate_policies(inp):
    validate(inp)
    n, m = len(inp["prior"]), len(inp["tests"])
    require(n <= 3 and m <= 2 and inp["initial_horizon"] <= 2, "policy enumeration restricted to <=3 hypotheses, <=2 tests, horizon<=2")
    p1 = [[Fraction(s) for s in row] for row in inp["p_one"]]
    costs = [Fraction(s) for s in inp["costs"]]
    terminals = [tuple(Fraction(s) for s in row) for row in inp["loss"]]
    w = [Fraction(s) for s in inp["prior"]]
    available = list(range(m))
    for event in inp["history"]:
        j, b = event["test"], event["outcome"]
        available.remove(j)
        w = [w[h] * (p1[j][h] if b else 1 - p1[j][h]) for h in range(n)]
    def trees(left, depth):
        policies = [(a, vec) for a, vec in enumerate(terminals)]
        if depth:
            for t in left:
                subtrees = trees(tuple(x for x in left if x != t), depth - 1)
                for (_, zero), (_, one) in product(subtrees, repeat=2):
                    vec = tuple(costs[t] + (1 - p1[t][h]) * zero[h] + p1[t][h] * one[h] for h in range(n))
                    policies.append((n + t, vec))
        return policies
    policies = trees(tuple(available), inp["initial_horizon"] - len(inp["history"]))
    if sum(w) == 0:
        return {"mass": "0/1", "values": [], "optimal": [], "policy_count": len(policies)}
    per_action = {}
    for action, vec in policies:
        risk = sum(a * b for a, b in zip(w, vec)) / sum(w)
        per_action[action] = min(per_action.get(action, risk), risk)
    best = min(per_action.values())
    return {"mass": qstr(sum(w)), "values": [[a, qstr(v)] for a, v in sorted(per_action.items())], "optimal": [a for a, v in sorted(per_action.items()) if v == best], "policy_count": len(policies)}
