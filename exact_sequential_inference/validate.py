"""Recompute artifacts, canonical group identities, model views, and invariants."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json

from . import SCHEMA_VERSION
from . import active, active_reference, temporal, temporal_reference
from .common import digest, dumps, integer, load_json, name, obj, require
from .generate import FIELDS, SPLITS, config_check, source_hashes
from .grouping import temporal_group
from .entropy_anchor import certificate
from .render import active_payload, temporal_payload, prompt
from .release_contract import ReleaseCoverage, validate_active_profile, validate_temporal_profile


def exact_equal(a, b, label):
    require(dumps(a) == dumps(b), label)  # True must never compare equal to 1.


def validate_record(r):
    obj(r, FIELDS, "record")
    require(r["schema_version"] == SCHEMA_VERSION, "record version")
    require(type(r["split"]) is str and r["split"] in SPLITS + ["anchors"], "split")
    require(type(r["mode"]) is str and r["mode"] in ("active_sensing", "noisy_temporal"), "mode")
    name(r["id"], "record id")
    # SHA256 IDs with prefixes can exceed 80 characters only for pair/temporal? all <=80.
    name(r["world_group_id"], "group id")
    name(r["paired_variant_id"], "pair id")
    require(type(r["audit"]) is dict, "audit object")
    audit = r["audit"]
    require(audit.get("renderer") in ("standard", "reserved_pretty"), "renderer")
    name(audit.get("variant"), "variant")
    require(audit.get("label_origin") == "exact_formal_computation", "label provenance")
    common_audit = {"renderer", "variant", "label_origin", "independent_oracle"}
    if r["mode"] == "active_sensing":
        extra = {"anchor", "entropy_order_certificate"} if r["split"] == "anchors" else {"prior_axis", "cost_axis", "loss_axis", "group_index"}
        obj(audit, common_audit | extra | {"optimal_actions"}, "active audit")
        require(audit["independent_oracle"] == "unnormalized_risk", "active reference name")
        if r["split"] != "anchors":
            integer(audit["prior_axis"], 0, 1, "prior axis")
            integer(audit["group_index"], 0, 19, "group index")
            require(audit["cost_axis"] in ("low", "high") and audit["loss_axis"] in ("symmetric", "skew"), "intervention axes")
        else:
            require(audit["anchor"] == "balanced_entropy_vs_loss_relevant_test", "anchor name")
            exact_equal(audit["entropy_order_certificate"], certificate(r["formal_input"]), "entropy anchor certificate")
    else:
        obj(audit, common_audit | {"hazard", "source_error", "family_index", "report_schedule", "latent_states_released"}, "temporal audit")
        require(audit["independent_oracle"] == "clamped_forward_transfer", "temporal reference name")
        require(audit["hazard"] in ("1/100", "2/25") and audit["source_error"] in ("0/1", "1/5"), "temporal release cell")
        integer(audit["family_index"], 0, 9, "family index")
        require(audit["latent_states_released"] is False, "latent labels must not be released")
        require(audit["report_schedule"] == "one_per_time_entity_t_mod16", "report schedule")
    inp, reports, queries = r["formal_input"], r["source_records"], r["queries"]
    if r["mode"] == "active_sensing":
        require(type(reports) is list and not reports and type(queries) is list and not queries, "active source/query fields must be empty arrays")
        active.validate(inp)
        validate_active_profile(r)
        target, trace, optimal = active.solve(inp)
        other, other_optimal = active_reference.solve(inp)
        exact_equal(target, other, "independent active target disagreement")
        exact_equal(optimal, other_optimal, "independent active tie disagreement")
        exact_equal(audit.get("optimal_actions"), optimal, "optimal set mismatch")
        group = active.group_id(inp["p_one"])
        rendered = {"decision": prompt(active_payload(inp), audit["renderer"])}
        pair = "pair_" + digest([group, inp["initial_horizon"]] if r["split"] == "anchors" else [group, inp["prior"], inp["history"]])
    else:
        temporal.validate(inp, reports, queries)
        validate_temporal_profile(r)
        target, trace = temporal.solve(inp, reports, queries)
        exact_equal(target, temporal_reference.solve(inp, reports, queries), "independent temporal target disagreement")
        group = temporal_group(inp, reports)
        pair = "pair_" + group
        rendered = {q["query_id"]: prompt(temporal_payload(inp, reports, q), audit["renderer"]) for q in queries}
    exact_equal(r["oracle_targets"], target, "target differs from exact oracle")
    exact_equal(r["exact_trace"], trace, "trace differs from exact recurrence")
    exact_equal(r["prompt"], rendered, "model prompt violates allowlist or cutoff")
    require(r["world_group_id"] == group, "canonical world group mismatch")
    require(r["paired_variant_id"] == pair, "pair id mismatch")
    identity = [r["mode"], group, r["split"], inp, reports, queries, audit["renderer"], audit["variant"]]
    require(r["id"] == "record_" + digest(identity), "record id mismatch")
    return True


def validate_dataset(directory, check_source=True):
    root = Path(directory)
    manifest = load_json(root / "manifest.json")
    coverage = ReleaseCoverage(config_check(manifest["config"]))
    record_path = root / "records.jsonl"
    require(hashlib.sha256(record_path.read_bytes()).hexdigest() == manifest["records_sha256"], "dataset hash mismatch")
    require(record_path.stat().st_size == manifest["bytes"], "dataset byte mismatch")
    groups, ids, pairs, counts, actions, statuses = {}, set(), {}, Counter(), Counter(), Counter()
    scalar_cost_invariance, duplicate_invariance = 0, 0
    smoothing_differences, high_cost_stops, low_cost_tests = 0, 0, 0
    cells, model_bytes = Counter(), 0
    with record_path.open(encoding="utf-8") as f:
        for line in f:
            # json duplicate key detection via one-object decoder, no loose parser.
            def unique(items):
                d = {}
                for k, v in items:
                    require(k not in d, "duplicate record JSON key")
                    d[k] = v
                return d
            r = json.loads(line, object_pairs_hook=unique)
            validate_record(r)
            coverage.add(r)
            require(r["id"] not in ids, "duplicate record id")
            ids.add(r["id"])
            g, split = r["world_group_id"], r["split"]
            require(g not in groups or groups[g] == split, "family leaked across splits")
            groups[g] = split
            key = r["mode"] + "/" + split
            counts[key] += 1
            model_bytes += sum(len(s.encode()) for s in r["prompt"].values())
            pair_key = r["paired_variant_id"]
            if r["mode"] == "active_sensing":
                target = r["oracle_targets"]
                signature = [target["status"], target["evidence_probability"], target["posterior"]]
                if pair_key in pairs:
                    exact_equal(pairs[pair_key], signature, "cost/loss intervention changed posterior")
                    scalar_cost_invariance += 1
                actions[str(target["selected_action"])] += 1
                statuses[target["status"]] += 1
                if target["selected_action"] is not None:
                    if r["audit"].get("cost_axis") == "high":
                        require(target["selected_action"] < 4, "high-cost anchor failed to stop")
                        high_cost_stops += 1
                    if r["audit"].get("cost_axis") == "low" and target["selected_action"] >= 4:
                        low_cost_tests += 1
            else:
                signature = r["oracle_targets"]
                if pair_key in pairs:
                    exact_equal(pairs[pair_key], signature, "duplicate/order intervention changed posterior")
                    duplicate_invariance += 1
                cells[r["audit"]["hazard"] + "/" + r["audit"]["source_error"]] += 1
                indexed = {t["query_id"]: t for t in signature}
                for i in (0, 1):
                    a = indexed.get(f"q{i}_historical_smoother_7_63")
                    b = indexed.get(f"q{i}_historical_as_known_7_7")
                    smoothing_differences += int(a is not None and b is not None and a["posterior"] != b["posterior"])
                statuses.update(t["status"] for t in signature)
            pairs[pair_key] = signature
    configured_slots = coverage.finish()
    exact_equal(dict(counts), manifest["counts"], "manifest split counts")
    exact_equal(groups, manifest["world_groups"], "manifest group map")
    require(len(ids) == manifest["records"], "manifest record total")
    if check_source:
        exact_equal(source_hashes(Path(__file__).resolve().parent.parent), manifest["source_sha256"], "source hash changed; retained artifact must be regenerated into a new directory")
    return {"release_profile_binding": "passed", "configured_group_slots": configured_slots, "records": len(ids), "world_groups": len(groups), "counts": dict(counts), "active_action_histogram": dict(actions), "status_histogram": dict(statuses), "temporal_cells": dict(cells), "cost_loss_pair_comparisons": scalar_cost_invariance, "duplicate_order_pair_comparisons": duplicate_invariance, "smoother_vs_as_known_differences": smoothing_differences, "high_cost_stop_records": high_cost_stops, "low_cost_sense_records": low_cost_tests, "model_prompt_bytes": model_bytes, "canonical_group_overlap": 0, "all_record_oracle_and_trace_checks": "passed"}
