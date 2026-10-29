"""Bind release labels to declared parameters, independently of label inference.

The mathematical solvers retain their bounded generic domains. These checks
apply to records claiming the v1 release's fixed experimental profile. Membership
in train/validation/test_id is assigned at dataset level, never inferred from a
parameter value shared by those three splits.
"""
from collections import Counter
from fractions import Fraction
from itertools import product
from .common import dumps, integer, qstr, require

ACTIVE_PRIORS = (["1/4"] * 4, ["1/10", "1/5", "3/10", "2/5"])
ACTIVE_HISTORIES = ([], [{"test": 0, "outcome": 0}],
                    [{"test": 0, "outcome": 1}, {"test": 1, "outcome": 0}],
                    [{"test": 2, "outcome": 1}, {"test": 0, "outcome": 0}, {"test": 3, "outcome": 1}])
ORDINARY_GRID = {"1/5", "2/5", "3/5", "4/5"}
OOD_GRID = {"0/1", "1/3", "2/3", "1/1"}
ANCHOR_TABLE = [["0/1", "0/1", "1/1", "1/1"], ["1/1", "0/1", "0/1", "0/1"], ["1/2"] * 4, ["1/2"] * 4]
ANCHOR_LOSS = [["0/1", "1/1", "1/1", "1/1"]] + [["8/1", "0/1", "0/1", "0/1"]] * 3


def equal(actual, expected, label):
    require(dumps(actual) == dumps(expected), "release binding: " + label)


def renderers(split):
    return ("standard", "reserved_pretty") if split == "ood_surface" else ("standard",)


def temporal_variants(split):
    ordinary = ("base", "duplicates", "reversed_duplicates")
    return ordinary + ("reserved_renderer",) if split == "ood_surface" else ordinary


def validate_active_profile(r):
    inp, audit, split = r["formal_input"], r["audit"], r["split"]
    require(len(inp["hypotheses"]) == 4 and len(inp["tests"]) == 4, "release binding: active domain is 4 hypotheses and 4 tests")
    namespace = "anchor" if split == "anchors" else ("reserved" if split == "ood_surface" else "item")
    equal(inp["hypotheses"], [f"{namespace}H{i}" for i in range(4)], "hypothesis namespace/split")
    equal(inp["tests"], [f"{namespace}T{i}" for i in range(4)], "test namespace/split")
    require(audit["renderer"] in renderers(split), "release binding: renderer/split")
    if split == "anchors":
        equal(inp["p_one"], ANCHOR_TABLE, "anchor likelihood table")
        equal(inp["prior"], ACTIVE_PRIORS[0], "anchor prior")
        equal(inp["loss"], ANCHOR_LOSS, "anchor loss")
        equal(inp["history"], [], "anchor history")
        require(inp["initial_horizon"] in (1, 3), "release binding: anchor horizon")
        cost = inp["costs"][0]
        require(cost in ("1/50", "3/1"), "release binding: anchor cost")
        equal(inp["costs"], [cost] * 4, "anchor cost vector")
        equal(audit["variant"], f"entropy_h{inp['initial_horizon']}_cost{cost.replace('/', '_')}", "anchor variant")
        return
    grid = OOD_GRID if split == "ood_likelihood" else ORDINARY_GRID
    require(all(p in grid for row in inp["p_one"] for p in row), "release binding: likelihood support/split")
    equal(inp["prior"], ACTIVE_PRIORS[audit["prior_axis"]], "prior_axis/prior")
    cost = "1/100" if audit["cost_axis"] == "low" else "3/1"
    equal(inp["costs"], [cost] * 4, "cost_axis/costs")
    loss = [[qstr(0 if a == h else (5 if audit["loss_axis"] == "skew" and h == 0 else 1)) for h in range(4)] for a in range(4)]
    equal(inp["loss"], loss, "loss_axis/loss")
    equal(inp["initial_horizon"], 3, "ordinary initial horizon")
    require(any(dumps(inp["history"]) == dumps(h) for h in ACTIVE_HISTORIES), "release binding: history support")
    equal(audit["variant"], f"{audit['cost_axis']}_{audit['loss_axis']}_{audit['renderer']}", "active variant axes")


def validate_temporal_profile(r):
    inp, audit, split = r["formal_input"], r["audit"], r["split"]
    require(split in ("train", "validation", "test_id", "ood_surface"), "release binding: temporal split")
    namespace = "reserved" if split == "ood_surface" else "entity"
    equal(inp["entities"], [f"{namespace}{i:02}" for i in range(16)], "temporal namespace/domain")
    equal(inp["steps"], 64, "temporal steps")
    equal(inp["prior"], ["1/2", "1/2"], "temporal prior")
    hazard, error = Fraction(audit["hazard"]), Fraction(audit["source_error"])
    equal(inp["transition"], [[qstr(1-hazard), qstr(hazard)], [qstr(hazard), qstr(1-hazard)]], "hazard/transition")
    expected_sources = [{"source_id": f"{namespace}source{i}", "emission": [[qstr(1-error), qstr(error)], [qstr(error), qstr(1-error)]]} for i in range(2)]
    equal(inp["sources"], expected_sources, "source_error/emissions and source namespace")
    variant = audit["variant"]
    require(variant in temporal_variants(split), "release binding: temporal variant/split")
    equal(audit["renderer"], "reserved_pretty" if variant == "reserved_renderer" else "standard", "temporal variant/renderer")
    expected_queries = []
    for i, entity in enumerate(inp["entities"][:2]):
        for kind, target, cutoff in (("current_filter", 31, 31), ("current_filter", 63, 63), ("historical_smoother", 7, 63), ("historical_as_known", 7, 7)):
            expected_queries.append({"query_id": f"q{i}_{kind}_{target}_{cutoff}", "kind": kind, "entity": entity, "target_time": target, "cutoff": cutoff})
    equal(r["queries"], expected_queries, "temporal release queries")
    # Distinct-origin cadence and the exact observable copy intervention are
    # checked directly, without relying on the seed or hidden trajectories.
    originals = [x for x in r["source_records"] if x["report_id"] == "copy_" + x["origin_report_id"]]
    require(len(originals) == 64 and {x["event_time"] for x in originals} == set(range(64)), "release binding: primary report cadence")
    for row in originals:
        t = row["event_time"]
        equal(row["origin_report_id"], f"origin{t % 16:02}_{t:02}", "public origin/time")
        equal(row["entity"], inp["entities"][t % 16], "cadence entity=t mod16")
        equal(row["source_id"], inp["sources"][(t // 16) % 2]["source_id"], "primary source schedule")
        require(row["arrival_time"] in {min(63, t+d) for d in (0, 1, 3)}, "release binding: primary delay support")
    base = sorted(originals, key=lambda x: (x["arrival_time"], x["report_id"]))
    expected = base
    if variant in ("duplicates", "reversed_duplicates"):
        copies = [dict(row, report_id="extra_"+row["origin_report_id"], arrival_time=min(63, row["arrival_time"]+2)) for row in base]
        expected = sorted(base + copies, key=lambda x: (x["arrival_time"], x["report_id"]))
        if variant == "reversed_duplicates":
            expected.reverse()
    equal(r["source_records"], expected, "temporal variant copy count/timing/order")


class ReleaseCoverage:
    """Require complete configured slots and their predeclared factorial variants.

    Seeded provenance is separately checked by full byte regeneration. A record
    alone cannot establish which exchangeable train/validation/test_id seed slot
    a world came from; this class binds indices and membership within a dataset.
    """
    def __init__(self, config):
        self.config = config
        self.slots = {}
        self.groups = {}
        self.variants = {}

    def add(self, r):
        audit, split = r["audit"], r["split"]
        if r["mode"] == "active_sensing":
            if split == "anchors":
                slot = ("active_sensing", "anchors")
                key = (r["formal_input"]["initial_horizon"], r["formal_input"]["costs"][0])
            else:
                require(split in self.config["active_splits"], "release binding: active split not configured")
                index = integer(audit["group_index"], 0, self.config["active_groups_per_split"]-1, "configured group index")
                slot = ("active_sensing", split, index)
                key = (audit["prior_axis"], dumps(r["formal_input"]["history"]), audit["cost_axis"], audit["loss_axis"], audit["renderer"])
        else:
            require(split in self.config["temporal_splits"], "release binding: temporal split not configured")
            index = integer(audit["family_index"], 0, self.config["temporal_families_per_cell_split"]-1, "configured family index")
            slot = ("noisy_temporal", split, audit["hazard"], audit["source_error"], index)
            key = audit["variant"]
        group = r["world_group_id"]
        require(slot not in self.slots or self.slots[slot] == group, "release binding: multiple worlds claim one configured slot")
        require(group not in self.groups or self.groups[group] == slot, "release binding: one world claims multiple configured slots")
        self.slots[slot], self.groups[group] = group, slot
        self.variants.setdefault(slot, Counter())[key] += 1

    def finish(self):
        expected = {("active_sensing", "anchors"): Counter(product((1, 3), ("1/50", "3/1")))}
        for split in self.config["active_splits"]:
            for index in range(self.config["active_groups_per_split"]):
                expected[("active_sensing", split, index)] = Counter(product(range(2), map(dumps, ACTIVE_HISTORIES), ("low", "high"), ("symmetric", "skew"), renderers(split)))
        for split in self.config["temporal_splits"]:
            for hazard, error, index in product(("1/100", "2/25"), ("0/1", "1/5"), range(self.config["temporal_families_per_cell_split"])):
                expected[("noisy_temporal", split, hazard, error, index)] = Counter(temporal_variants(split))
        require(self.variants == expected, "release binding: missing/duplicate configured world slots or factorial variants")
        return len(expected)
