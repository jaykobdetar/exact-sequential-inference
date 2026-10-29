"""Exact event-time forward/backward inference under an arrival-time cutoff.

General declared transition/emission matrices are supported. The release config
selects the architect-approved symmetric hazard/error cells.
"""
from fractions import Fraction
from .common import array, integer, name, names, obj, qstr, require, vector

KINDS = {"current_filter", "historical_smoother", "historical_as_known"}


def validate(inp, reports, queries):
    obj(inp, {"entities", "steps", "prior", "transition", "sources"}, "temporal input")
    names(inp["entities"], 1, 16, "entities")
    steps = integer(inp["steps"], 1, 64, "steps")
    vector(inp["prior"], 2, True, True)
    array(inp["transition"], 2, 2, "transition")
    for row in inp["transition"]:
        vector(row, 2, True, True)
    array(inp["sources"], 1, 4, "sources")
    source_ids = []
    for source in inp["sources"]:
        obj(source, {"source_id", "emission"}, "source")
        source_ids.append(name(source["source_id"], "source id"))
        array(source["emission"], 2, 2, "emission")
        for row in source["emission"]:
            vector(row, 2, True, True)
    require(len(set(source_ids)) == len(source_ids), "duplicate source id")
    array(reports, 0, 8192, "reports")
    ids, origins = set(), {}
    for report in reports:
        obj(report, {"report_id", "origin_report_id", "entity", "event_time", "arrival_time", "source_id", "value"}, "report")
        rid = name(report["report_id"], "report id")
        origin = name(report["origin_report_id"], "origin id")
        require(rid not in ids, "duplicate report id")
        ids.add(rid)
        require(type(report["entity"]) is str and report["entity"] in inp["entities"], "unknown entity")
        require(type(report["source_id"]) is str and report["source_id"] in source_ids, "unknown source")
        event = integer(report["event_time"], 0, steps - 1, "event time")
        integer(report["arrival_time"], event, steps - 1, "arrival time")
        integer(report["value"], 0, 1, "reported value")
        payload = (report["entity"], event, report["source_id"], report["value"])
        require(origin not in origins or origins[origin] == payload, "conflicting copies of one origin")
        origins[origin] = payload
    array(queries, 1, 64, "queries")
    qids = set()
    for query in queries:
        obj(query, {"query_id", "kind", "entity", "target_time", "cutoff"}, "query")
        qid = name(query["query_id"], "query id")
        require(qid not in qids, "duplicate query id")
        qids.add(qid)
        require(type(query["kind"]) is str and query["kind"] in KINDS, "unknown query kind")
        require(type(query["entity"]) is str and query["entity"] in inp["entities"], "unknown query entity")
        cutoff = integer(query["cutoff"], 0, steps - 1, "cutoff")
        target = integer(query["target_time"], 0, cutoff, "target time")
        if query["kind"] != "historical_smoother":
            require(target == cutoff, "filter/as-known queries require target_time=cutoff")
    return inp


def available_origins(reports, cutoff):
    unique = {}
    for report in reports:
        if report["arrival_time"] <= cutoff:
            unique.setdefault(report["origin_report_id"], report)
    return [unique[k] for k in sorted(unique)]


def solve(inp, reports, queries):
    validate(inp, reports, queries)
    transition = [[Fraction(x) for x in row] for row in inp["transition"]]
    sources = {s["source_id"]: [[Fraction(x) for x in row] for row in s["emission"]] for s in inp["sources"]}
    prior = list(map(Fraction, inp["prior"]))
    cache, targets = {}, []
    requested = {(q["entity"], q["cutoff"]) for q in queries}
    # Each cutoff uses only records available at that cutoff; every entity is
    # checked, because zero mass elsewhere makes the joint conditioning invalid.
    for cutoff in sorted({q["cutoff"] for q in queries}):
        selected = available_origins(reports, cutoff)
        entities = {}
        global_mass = Fraction(1)
        for entity in inp["entities"]:
            emissions = [[Fraction(1), Fraction(1)] for _ in range(cutoff + 1)]
            origin_ids = []
            for report in selected:
                if report["entity"] == entity:
                    origin_ids.append(report["origin_report_id"])
                    row = emissions[report["event_time"]]
                    for state in (0, 1):
                        row[state] *= sources[report["source_id"]][state][report["value"]]
            alpha = [[prior[x] * emissions[0][x] for x in (0, 1)]]
            for t in range(1, cutoff + 1):
                alpha.append([emissions[t][x] * sum(alpha[-1][y] * transition[y][x] for y in (0, 1)) for x in (0, 1)])
            mass = sum(alpha[-1])
            global_mass *= mass
            # Beta starts at 1: there are no observations after cutoff.
            beta = [[Fraction(1), Fraction(1)] for _ in range(cutoff + 1)]
            for t in range(cutoff - 1, -1, -1):
                beta[t] = [sum(transition[x][y] * emissions[t + 1][y] * beta[t + 1][y] for y in (0, 1)) for x in (0, 1)]
            entities[entity] = {"origin_ids": origin_ids, "mass": qstr(mass)}
            if (entity, cutoff) in requested:
                entities[entity].update({"emissions": [[qstr(x) for x in row] for row in emissions], "alpha": [[qstr(x) for x in row] for row in alpha], "beta": [[qstr(x) for x in row] for row in beta]})
        cache[str(cutoff)] = {"global_mass": qstr(global_mass), "entities": entities}
    for query in queries:
        bank = cache[str(query["cutoff"])]
        e = bank["entities"][query["entity"]]
        target = query["target_time"]
        global_mass = Fraction(bank["global_mass"])
        posterior = None
        if global_mass:
            weights = [Fraction(e["alpha"][target][x]) * Fraction(e["beta"][target][x]) for x in (0, 1)]
            posterior = [qstr(x / sum(weights)) for x in weights]
        targets.append({"query_id": query["query_id"], "status": "ok" if global_mass else "impossible_evidence", "posterior": posterior, "global_evidence_probability": bank["global_mass"], "entity_evidence_probability": e["mass"]})
    return targets, {"cutoffs": cache}
