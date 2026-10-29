"""Posterior-normalized finite-horizon Bellman inference; all arithmetic exact."""
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from .common import array, digest, dumps, integer, names, obj, qstr, require, vector


def validate(inp):
    obj(inp, {"hypotheses", "tests", "prior", "p_one", "costs", "loss", "initial_horizon", "history"}, "active input")
    n = len(names(inp["hypotheses"], 2, 4, "hypotheses"))
    m = len(names(inp["tests"], 1, 4, "tests"))
    vector(inp["prior"], n, True, True)
    array(inp["p_one"], m, m, "likelihood table")
    for row in inp["p_one"]:
        vector(row, n, True)
    vector(inp["costs"], m)
    array(inp["loss"], n, n, "terminal loss")
    for row in inp["loss"]:
        vector(row, n)
    horizon = integer(inp["initial_horizon"], 0, min(3, m), "initial horizon")
    array(inp["history"], 0, horizon, "history")
    used = set()
    for row in inp["history"]:
        obj(row, {"test", "outcome"}, "history item")
        t = integer(row["test"], 0, m - 1, "test")
        integer(row["outcome"], 0, 1, "outcome")
        require(t not in used, "test reused")
        used.add(t)
    return inp


def canonical_table(p_one):
    """Exact orbit key modulo common hypothesis permutation, tests and binary flips.

    Once a hypothesis order is fixed, column complement and test sorting minimize
    those independent permutations. Only <=24 hypothesis orders remain.
    """
    m = len(array(p_one, 1, 4, "table"))
    n = len(array(p_one[0], 2, 4, "table row"))
    cols = [vector(row, n, True) for row in p_one]
    forms = []
    for perm in permutations(range(n)):
        oriented = []
        for col in cols:
            a = tuple(qstr(col[h]) for h in perm)
            b = tuple(qstr(1 - col[h]) for h in perm)
            oriented.append(min(a, b))
        forms.append(dumps([n, m, sorted(oriented)]))
    return min(forms)


def group_id(p_one):
    return "active_" + digest(canonical_table(p_one))


def solve(inp):
    validate(inp)
    n, m = len(inp["prior"]), len(inp["tests"])
    prior = tuple(map(Fraction, inp["prior"]))
    one = tuple(tuple(map(Fraction, row)) for row in inp["p_one"])
    costs = tuple(map(Fraction, inp["costs"]))
    loss = tuple(tuple(map(Fraction, row)) for row in inp["loss"])
    history = tuple((row["test"], row["outcome"]) for row in inp["history"])
    nodes = {}

    @lru_cache(None)
    def visit(hist):
        weights = list(prior)
        for t, b in hist:
            weights = [w * (one[t][h] if b else 1 - one[t][h]) for h, w in enumerate(weights)]
        mass = sum(weights)
        key = ";".join(f"{t}:{b}" for t, b in hist) or "root"
        remaining = inp["initial_horizon"] - len(hist)
        if mass == 0:
            nodes[key] = {"history": [list(x) for x in hist], "remaining": remaining, "mass": "0/1", "posterior": None, "values": [], "best": None, "optimal": []}
            return None
        posterior = tuple(w / mass for w in weights)
        values = [(a, sum(p * v for p, v in zip(posterior, row))) for a, row in enumerate(loss)]
        if remaining:
            used = {t for t, b in hist}
            for t in range(m):
                if t in used:
                    continue
                cost = costs[t]
                for b in (0, 1):
                    prob = sum(p * (one[t][h] if b else 1 - one[t][h]) for h, p in enumerate(posterior))
                    child = visit(hist + ((t, b),))
                    if prob:
                        cost += prob * child
                values.append((n + t, cost))
        best = min(v for a, v in values)
        optimal = [a for a, v in values if v == best]
        nodes[key] = {"history": [list(x) for x in hist], "remaining": remaining, "mass": qstr(mass), "posterior": list(map(qstr, posterior)), "values": [[a, qstr(v)] for a, v in values], "best": qstr(best), "optimal": optimal}
        return best

    visit(history)
    key = ";".join(f"{t}:{b}" for t, b in history) or "root"
    root = nodes[key]
    target = {"status": "ok" if root["posterior"] is not None else "impossible_evidence", "evidence_probability": root["mass"], "posterior": root["posterior"], "action_values": root["values"], "minimum_expected_loss": root["best"], "selected_action": min(root["optimal"]) if root["optimal"] else None}
    trace = {"action_order": {"guesses": list(range(n)), "tests": list(range(n, n + m))}, "root": key, "nodes": [nodes[k] for k in sorted(nodes)]}
    return target, trace, root["optimal"]
