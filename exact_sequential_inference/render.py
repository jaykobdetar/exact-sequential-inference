"""Fixed implementation-authored templates. No oracle values enter model inputs."""
from copy import deepcopy
from .common import dumps

REPORT_FIELDS = ("report_id", "origin_report_id", "entity", "event_time", "arrival_time", "source_id", "value")
TEMPORAL_FIELDS = ("entities", "steps", "prior", "transition", "sources")

ACTIVE_RULES = "Hidden hypotheses have the declared prior. Tests are binary, conditionally independent given the hypothesis, and each test may be used once. p_one[test][hypothesis] is P(outcome=1). initial_horizon includes tests in history. A test pays its cost immediately, reveals its outcome, and consumes one remaining test opportunity. You may instead stop and guess; at zero remaining horizon you must guess. loss[guess][hypothesis] is terminal loss. Minimize future total expected loss, excluding sunk costs. Action indices are guesses 0..H-1 followed by tests H..H+T-1. Return the exact posterior and a minimum-loss action; the canonical answer uses the smallest optimal action index. Impossible evidence has no posterior or optimal action."
TEMPORAL_RULES = "Each entity has an independent binary Markov chain. The declared prior is for X_0; transition[old][new] applies once between successive event times. emission[state][reported_bit] is declared for each source. Distinct origin_report_id emissions are independent conditional on latent states; copies of one origin count once as soon as any copy arrives. Event time states what time a report concerns; arrival time states when it becomes available. Only reports with arrival_time<=query.cutoff are supplied. current_filter asks P(X_cutoff | reports available by cutoff); historical_smoother asks P(X_target_time | reports available by cutoff); historical_as_known asks P(X_target_time | reports available by target_time), so its cutoff equals target_time. Return an exact two-state posterior. If the global available evidence has probability zero, return impossible_evidence."


def active_payload(inp, renderer="standard"):
    # Explicit allowlist, rather than serializing the parent record.
    return {"instructions": ACTIVE_RULES, "formal_input": deepcopy({k: inp[k] for k in ("hypotheses", "tests", "prior", "p_one", "costs", "loss", "initial_horizon", "history")})}


def temporal_payload(inp, reports, query, renderer="standard"):
    return {"instructions": TEMPORAL_RULES, "formal_input": deepcopy({k: inp[k] for k in ("entities", "steps", "prior", "transition", "sources")}), "source_records": deepcopy([{k: r[k] for k in REPORT_FIELDS} for r in reports if r["arrival_time"] <= query["cutoff"]]), "query": deepcopy({k: query[k] for k in ("kind", "entity", "target_time", "cutoff")})}


def prompt(payload, renderer):
    if renderer == "standard":
        return dumps(payload)
    if renderer == "reserved_pretty":
        import json
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    raise ValueError("unknown renderer")


def adaptation_payload(inp, reports, cutoff):
    """Source-only adaptation export: no questions, targets, traces or latent states."""
    return {"formal_input": deepcopy({k: inp[k] for k in TEMPORAL_FIELDS}), "cutoff": cutoff, "source_records": deepcopy([{k: r[k] for k in REPORT_FIELDS} for r in reports if r["arrival_time"] <= cutoff])}
