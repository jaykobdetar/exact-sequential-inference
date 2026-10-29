"""Family identities for entire temporal streams, independent of display names."""
from .common import digest, dumps


def temporal_group(inp, reports):
    # Earliest arrival introduces an origin. Extra copies never create evidence.
    origins = {}
    for r in reports:
        old = origins.get(r["origin_report_id"])
        if old is None or r["arrival_time"] < old["arrival_time"]:
            origins[r["origin_report_id"]] = r
    sources = {s["source_id"]: s["emission"] for s in inp["sources"]}
    chains = []
    for entity in inp["entities"]:
        events = [[r["event_time"], r["arrival_time"], r["value"], sources[r["source_id"]]] for r in origins.values() if r["entity"] == entity]
        chains.append(sorted(events, key=dumps))
    semantic = {"steps": inp["steps"], "prior": inp["prior"], "transition": inp["transition"], "chains": sorted(chains, key=dumps)}
    return "temporal_" + digest(semantic)
