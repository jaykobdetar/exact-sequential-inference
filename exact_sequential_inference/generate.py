"""Seeded small-release generation. Single process; no learning or model calls."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import hashlib
import random

from . import SCHEMA_VERSION, __version__
from . import active, active_reference, temporal, temporal_reference
from .common import array, digest, dumps, integer, obj, qstr, require
from .grouping import temporal_group
from .entropy_anchor import certificate
from .render import active_payload, temporal_payload, prompt

SPLITS = ["train", "validation", "test_id", "ood_likelihood", "ood_surface"]
FIELDS = {"schema_version", "id", "world_group_id", "split", "mode", "formal_input", "source_records", "queries", "prompt", "oracle_targets", "exact_trace", "paired_variant_id", "audit"}


def config_check(c):
    obj(c, {"schema_version", "seed", "active_groups_per_split", "temporal_families_per_cell_split", "active_splits", "temporal_splits", "max_dataset_bytes", "temporal_schedule"}, "config")
    require(c["schema_version"] == "exact_sequential_inference-config/1.0", "config version")
    require(c["temporal_schedule"] == "one_per_time_entity_t_mod16", "primary temporal cadence is fixed")
    integer(c["seed"], 0, 2**63 - 1, "seed")
    integer(c["active_groups_per_split"], 1, 20, "active groups per split")
    integer(c["temporal_families_per_cell_split"], 1, 10, "temporal families per cell/split")
    integer(c["max_dataset_bytes"], 1000, 20000000, "dataset byte cap")
    for key in ("active_splits", "temporal_splits"):
        array(c[key], 1, 5, key)
        require(all(type(s) is str and s in SPLITS for s in c[key]), "unknown split")
        require(len(set(c[key])) == len(c[key]), "duplicate splits")
    require("ood_likelihood" not in c["temporal_splits"], "likelihood OOD is an active-only axis")
    return c


def rng_for(seed, key):
    return random.Random(int(digest([seed, key]), 16))


def record(mode, group, split, inp, reports, queries, renderer, variant, pair, audit):
    if mode == "active_sensing":
        target, trace, optimal = active.solve(inp)
        reference, reference_optimal = active_reference.solve(inp)
        require(target == reference and optimal == reference_optimal, "active oracle disagreement")
        rendered = {"decision": prompt(active_payload(inp), renderer)}
        audit = {**audit, "optimal_actions": optimal, "independent_oracle": "unnormalized_risk"}
    else:
        target, trace = temporal.solve(inp, reports, queries)
        require(target == temporal_reference.solve(inp, reports, queries), "temporal oracle disagreement")
        rendered = {q["query_id"]: prompt(temporal_payload(inp, reports, q), renderer) for q in queries}
        audit = {**audit, "independent_oracle": "clamped_forward_transfer"}
    identity = [mode, group, split, inp, reports, queries, renderer, variant]
    return {"schema_version": SCHEMA_VERSION, "id": "record_" + digest(identity), "world_group_id": group, "split": split, "mode": mode, "formal_input": inp, "source_records": reports, "queries": queries, "prompt": rendered, "oracle_targets": target, "exact_trace": trace, "paired_variant_id": pair, "audit": {**audit, "renderer": renderer, "variant": variant, "label_origin": "exact_formal_computation"}}


def active_records(config):
    occupied = set()
    for split in config["active_splits"]:
        for group_index in range(config["active_groups_per_split"]):
            rng = rng_for(config["seed"], ["active", split, group_index])
            while True:
                grid = [Fraction(i, 3) for i in range(4)] if split == "ood_likelihood" else [Fraction(i, 5) for i in range(1, 5)]
                table = [[qstr(rng.choice(grid)) for _ in range(4)] for _ in range(4)]
                group = active.group_id(table)
                if group not in occupied:
                    occupied.add(group)
                    break
            namespace = "reserved" if split == "ood_surface" else "item"
            renderers = ["standard", "reserved_pretty"] if split == "ood_surface" else ["standard"]
            priors = [["1/4"] * 4, ["1/10", "1/5", "3/10", "2/5"]]
            histories = [[], [{"test": 0, "outcome": 0}], [{"test": 0, "outcome": 1}, {"test": 1, "outcome": 0}], [{"test": 2, "outcome": 1}, {"test": 0, "outcome": 0}, {"test": 3, "outcome": 1}]]
            for prior_idx, prior in enumerate(priors):
                for history in histories:
                    pair = "pair_" + digest([group, prior, history])
                    for cost_name, cost in [("low", "1/100"), ("high", "3/1")]:
                        for loss_name in ("symmetric", "skew"):
                            loss = [[qstr(0 if a == h else (5 if loss_name == "skew" and h == 0 else 1)) for h in range(4)] for a in range(4)]
                            inp = {"hypotheses": [f"{namespace}H{x}" for x in range(4)], "tests": [f"{namespace}T{x}" for x in range(4)], "prior": prior, "p_one": table, "costs": [cost] * 4, "loss": loss, "initial_horizon": 3, "history": history}
                            for renderer in renderers:
                                variant = f"{cost_name}_{loss_name}_{renderer}"
                                yield record("active_sensing", group, split, inp, [], [], renderer, variant, pair, {"prior_axis": prior_idx, "cost_axis": cost_name, "loss_axis": loss_name, "group_index": group_index})
    # Predeclared anchor: entropy's balanced partition differs from the test that
    # perfectly determines the loss-relevant event H=0. Stored outside train/test.
    table = [["0/1", "0/1", "1/1", "1/1"], ["1/1", "0/1", "0/1", "0/1"], ["1/2"] * 4, ["1/2"] * 4]
    group = active.group_id(table)
    require(group not in occupied, "anchor collided with generated table")
    for horizon in (1, 3):
        for cost in ("1/50", "3/1"):
            inp = {"hypotheses": [f"anchorH{x}" for x in range(4)], "tests": [f"anchorT{x}" for x in range(4)], "prior": ["1/4"] * 4, "p_one": table, "costs": [cost] * 4, "loss": [["0/1", "1/1", "1/1", "1/1"], ["8/1", "0/1", "0/1", "0/1"], ["8/1", "0/1", "0/1", "0/1"], ["8/1", "0/1", "0/1", "0/1"]], "initial_horizon": horizon, "history": []}
            yield record("active_sensing", group, "anchors", inp, [], [], "standard", f"entropy_h{horizon}_cost{cost.replace('/', '_')}", "pair_" + digest([group, horizon]), {"anchor": "balanced_entropy_vs_loss_relevant_test", "entropy_order_certificate": certificate(inp)})


def coin(rng, p):
    return int(rng.randrange(p.denominator) < p.numerator)


def temporal_records(config):
    occupied = set()
    for split in config["temporal_splits"]:
        for hazard in (Fraction(1, 100), Fraction(2, 25)):
            for error in (Fraction(0), Fraction(1, 5)):
                for index in range(config["temporal_families_per_cell_split"]):
                    seed_key = ["temporal", split, qstr(hazard), qstr(error), index]
                    rng = rng_for(config["seed"], seed_key)
                    delays = rng_for(config["seed"], [seed_key, "exogenous_delays"])
                    namespace = "reserved" if split == "ood_surface" else "entity"
                    inp = {"entities": [f"{namespace}{i:02}" for i in range(16)], "steps": 64, "prior": ["1/2", "1/2"], "transition": [[qstr(1 - hazard), qstr(hazard)], [qstr(hazard), qstr(1 - hazard)]], "sources": [{"source_id": f"{namespace}source{i}", "emission": [[qstr(1 - error), qstr(error)], [qstr(error), qstr(1 - error)]]} for i in range(2)]}
                    reports = []
                    states = [coin(rng, Fraction(1, 2)) for _ in range(16)]
                    for t in range(64):
                        if t:
                            states = [state ^ coin(rng, hazard) for state in states]
                        entity_index = t % 16
                        entity = inp["entities"][entity_index]
                        origin = f"origin{entity_index:02}_{t:02}"
                        reports.append({"report_id": "copy_" + origin, "origin_report_id": origin, "entity": entity, "event_time": t, "arrival_time": min(63, t + delays.choice((0, 1, 3))), "source_id": inp["sources"][(t // 16) % 2]["source_id"], "value": states[entity_index] ^ coin(rng, error)})
                    reports.sort(key=lambda r: (r["arrival_time"], r["report_id"]))
                    group = temporal_group(inp, reports)
                    require(group not in occupied, "temporal stream collision; choose a new seed, never resplit")
                    occupied.add(group)
                    queries = []
                    for i, entity in enumerate(inp["entities"][:2]):
                        for kind, target, cutoff in [("current_filter", 31, 31), ("current_filter", 63, 63), ("historical_smoother", 7, 63), ("historical_as_known", 7, 7)]:
                            queries.append({"query_id": f"q{i}_{kind}_{target}_{cutoff}", "kind": kind, "entity": entity, "target_time": target, "cutoff": cutoff})
                    duplicates = deepcopy(reports)
                    for r in reports:
                        copy = dict(r, report_id="extra_" + r["origin_report_id"], arrival_time=min(63, r["arrival_time"] + 2))
                        duplicates.append(copy)
                    duplicate_ordered = sorted(duplicates, key=lambda r: (r["arrival_time"], r["report_id"]))
                    variants = [("base", reports), ("duplicates", duplicate_ordered), ("reversed_duplicates", list(reversed(duplicate_ordered)))]
                    if split == "ood_surface":
                        variants.append(("reserved_renderer", reports))
                    for variant, source_records in variants:
                        renderer = "reserved_pretty" if variant == "reserved_renderer" else "standard"
                        require(temporal_group(inp, source_records) == group, "paired variant changed semantic family")
                        yield record("noisy_temporal", group, split, inp, source_records, queries, renderer, variant, "pair_" + group, {"hazard": qstr(hazard), "source_error": qstr(error), "family_index": index, "report_schedule": config["temporal_schedule"], "latent_states_released": False})


def source_hashes(root):
    paths = []
    for folder, suffix in [("exact_sequential_inference", ".py"), ("schemas", ".json"), ("configs", ".json"), ("evidence", ".json"), ("tests", ".py"), ("scripts", ".py")]:
        paths += list((root / folder).glob("*" + suffix))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def generate(config, output):
    config_check(config)
    output = Path(output)
    require(not output.exists(), "output directory already exists; use a new path to preserve evidence")
    output.mkdir(parents=True)
    path = output / "records.jsonl"
    counts, groups, byte_count = {}, {}, 0
    with path.open("w", encoding="utf-8") as f:
        from itertools import chain
        for r in chain(active_records(config), temporal_records(config)):
            line = dumps(r) + "\n"
            byte_count += len(line.encode())
            require(byte_count <= config["max_dataset_bytes"], "dataset byte cap exceeded; partial output retained")
            f.write(line)
            key = r["mode"] + "/" + r["split"]
            counts[key] = counts.get(key, 0) + 1
            require(r["world_group_id"] not in groups or groups[r["world_group_id"]] == r["split"], "cross-split group collision")
            groups[r["world_group_id"]] = r["split"]
    root = Path(__file__).resolve().parent.parent
    manifest = {"schema_version": SCHEMA_VERSION, "generator_version": __version__, "historical_evidence_status": "not_bundled", "config": config, "records": sum(counts.values()), "counts": counts, "world_groups": groups, "bytes": byte_count, "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_sha256": source_hashes(root), "model_input_allowlist": {"active_sensing": ["instructions", "formal_input"], "noisy_temporal": ["instructions", "formal_input", "source_records filtered by query cutoff", "query without query_id"]}, "adaptation_allowlist": ["formal_input", "cutoff", "source_records filtered by cutoff"], "label_provenance": "Exact rational formal engines cross-checked by independently implemented exact reference engines. No model outputs or model calls.", "split_unit": "whole canonical likelihood table or whole canonical temporal stream, including all pairs and renderings"}
    (output / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    return manifest
