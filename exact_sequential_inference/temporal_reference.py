"""Independent variable-elimination oracle and tiny latent-path enumeration.

The scalable oracle clamps the target state and sums with a forward transfer
recurrence; it never uses a backward pass. The <=8-step oracle enumerates all
2^T latent paths per entity. Deduplication is implemented independently.
"""
from fractions import Fraction
from itertools import product
from .common import qstr, require
from .temporal import validate


def solve(inp, reports, queries, exhaustive=False):
    validate(inp, reports, queries)
    if exhaustive:
        require(inp["steps"] <= 8, "path enumeration limited to <=8 steps")
    prior = list(map(Fraction, inp["prior"]))
    transition = [[Fraction(x) for x in row] for row in inp["transition"]]
    sources = {s["source_id"]: [[Fraction(x) for x in row] for row in s["emission"]] for s in inp["sources"]}
    cache = {}
    def mass(entity, cutoff, clamp=None):
        key = entity, cutoff, clamp
        if key in cache:
            return cache[key]
        eligible = {}
        for row in reversed(reports):
            if row["entity"] == entity and row["arrival_time"] <= cutoff:
                eligible[row["origin_report_id"]] = row
        by_time = [[] for _ in range(cutoff + 1)]
        for row in eligible.values():
            by_time[row["event_time"]].append(row)
        def emission(t, x):
            out = Fraction(1)
            for row in by_time[t]:
                out *= sources[row["source_id"]][x][row["value"]]
            return out
        if exhaustive:
            result = Fraction(0)
            for path in product((0, 1), repeat=cutoff + 1):
                if clamp is not None and path[clamp[0]] != clamp[1]:
                    continue
                prob = prior[path[0]]
                for t, state in enumerate(path):
                    prob *= emission(t, state)
                    if t:
                        prob *= transition[path[t - 1]][state]
                result += prob
        else:
            weights = prior[:]
            for t in range(cutoff + 1):
                if t:
                    weights = [sum(weights[y] * transition[y][x] for y in (0, 1)) for x in (0, 1)]
                weights = [weights[x] * emission(t, x) if clamp is None or clamp[0] != t or clamp[1] == x else Fraction(0) for x in (0, 1)]
            result = sum(weights)
        cache[key] = result
        return result
    result = []
    for query in queries:
        entity, cutoff, target = query["entity"], query["cutoff"], query["target_time"]
        global_mass = Fraction(1)
        for other in inp["entities"]:
            global_mass *= mass(other, cutoff)
        entity_mass = mass(entity, cutoff)
        posterior = [qstr(mass(entity, cutoff, (target, x)) / entity_mass) for x in (0, 1)] if global_mass else None
        result.append({"query_id": query["query_id"], "status": "ok" if global_mass else "impossible_evidence", "posterior": posterior, "global_evidence_probability": qstr(global_mass), "entity_evidence_probability": qstr(entity_mass)})
    return result
