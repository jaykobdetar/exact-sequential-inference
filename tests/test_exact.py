import copy
from fractions import Fraction
from itertools import permutations, product
import json
import unittest

from exact_sequential_inference import active, active_reference, temporal, temporal_reference
from exact_sequential_inference.common import InvalidInput, dumps, qstr, rational
from exact_sequential_inference.grouping import temporal_group
from exact_sequential_inference.render import temporal_payload, adaptation_payload
from exact_sequential_inference.scoring import score_active, score_temporal


def tiny_active(n=2, m=2, horizon=2):
    return {"hypotheses": [f"H{i}" for i in range(n)], "tests": [f"T{i}" for i in range(m)], "prior": [qstr(Fraction(1, n))] * n, "p_one": [["1/2"] * n for _ in range(m)], "costs": ["0/1"] * m, "loss": [[qstr(int(a != h)) for h in range(n)] for a in range(n)], "initial_horizon": horizon, "history": []}


def tiny_temporal(steps=3, entities=1, hazard="1/4", error="1/5"):
    h, e = Fraction(hazard), Fraction(error)
    return {"entities": [f"E{i}" for i in range(entities)], "steps": steps, "prior": ["1/2", "1/2"], "transition": [[qstr(1-h), qstr(h)], [qstr(h), qstr(1-h)]], "sources": [{"source_id": "S0", "emission": [[qstr(1-e), qstr(e)], [qstr(e), qstr(1-e)]]}]}


def report(t=0, b=1, arrival=None, origin="origin", entity="E0", rid=None):
    return {"report_id": rid or "copy_"+origin, "origin_report_id": origin, "entity": entity, "event_time": t, "arrival_time": t if arrival is None else arrival, "source_id": "S0", "value": b}


def query(t=0, cutoff=2, kind="historical_smoother", entity="E0"):
    return {"query_id": "query", "kind": kind, "entity": entity, "target_time": t, "cutoff": cutoff}


class CodecTests(unittest.TestCase):
    def test_exact_normalization_and_malformed(self):
        self.assertEqual(qstr(Fraction(4, 6)), "2/3")
        self.assertEqual(rational("0/1", True), 0)
        for v in [True, False, 1, 1.0, None, [], {}, "0", "0.5", "2/4", "0/2", "-1/2", "1/-2", "1/0", "01/2", "1/02", " 1/2", "1/2 ", "1e3/2", "NaN", "1/1\n", "9"*4096+"/1"]:
            with self.subTest(value=repr(v)[:80]), self.assertRaises(InvalidInput):
                rational(v, True)
        with self.assertRaises(InvalidInput):
            rational("2/1", True)


class ActiveTests(unittest.TestCase):
    def assert_oracles(self, inp, enumerate_trees=True):
        actual, trace, optimal = active.solve(inp)
        other, opts = active_reference.solve(inp)
        self.assertEqual(dumps(actual), dumps(other))
        self.assertEqual(optimal, opts)
        if enumerate_trees:
            exhaustive = active_reference.enumerate_policies(inp)
            self.assertEqual(actual["evidence_probability"], exhaustive["mass"])
            self.assertEqual(actual["action_values"], exhaustive["values"])
            self.assertEqual(optimal, exhaustive["optimal"])
        return actual, trace, optimal

    def test_exhaustive_two_state_two_test_policy_trees(self):
        # Exactly 3^4 * 3 * 2 = 486 input worlds, each all 74 h=2 policies.
        cases = 0
        for cells in product(("0/1", "1/2", "1/1"), repeat=4):
            for prior in (["0/1", "1/1"], ["1/2", "1/2"], ["1/1", "0/1"]):
                for cost in ("0/1", "1/4"):
                    inp = tiny_active()
                    inp.update(p_one=[list(cells[:2]), list(cells[2:])], prior=prior, costs=[cost]*2)
                    self.assert_oracles(inp)
                    cases += 1
        self.assertEqual(cases, 486)

    def test_exhaustive_three_state_deterministic_tables(self):
        for cells in product(("0/1", "1/1"), repeat=6):
            inp = tiny_active(3)
            inp.update(p_one=[list(cells[:3]), list(cells[3:])], costs=["1/10"]*2)
            self.assert_oracles(inp)

    def test_all_ordered_histories_and_terminal(self):
        inp = tiny_active()
        inp["p_one"] = [["0/1", "1/1"], ["1/2", "1/1"]]
        count = 0
        for depth in range(3):
            for tests in permutations(range(2), depth):
                for bits in product((0, 1), repeat=depth):
                    inp["history"] = [{"test": t, "outcome": b} for t, b in zip(tests, bits)]
                    answer, trace, _ = self.assert_oracles(inp)
                    if depth == 2:
                        self.assertTrue(all(a < 2 for a, _ in answer["action_values"]))
                    count += 1
        self.assertEqual(count, 13)

    def test_four_state_bound_ties_cost_independence(self):
        inp = tiny_active(4, 4, 3)
        target, trace, opts = self.assert_oracles(inp, False)
        self.assertEqual(len(trace["nodes"]), 249)
        self.assertEqual(opts, list(range(8)))
        self.assertEqual(target["selected_action"], 0)
        alternative = {"status": "ok", "posterior": target["posterior"], "action": 7}
        self.assertTrue(score_active(alternative, target, opts)["correct"])
        alternative["action"] = True
        with self.assertRaises(InvalidInput):
            score_active(alternative, target, opts)
        inp["costs"] = ["9/1"]*4
        new, _, _ = self.assert_oracles(inp, False)
        self.assertEqual(target["posterior"], new["posterior"])
        self.assertEqual(new["selected_action"], 0)

    def test_entropy_disagreement_hand_anchor(self):
        inp = tiny_active(4, 4, 1)
        inp["p_one"] = [["0/1","0/1","1/1","1/1"], ["1/1","0/1","0/1","0/1"], ["1/2"]*4, ["1/2"]*4]
        inp["costs"] = ["1/50"]*4
        inp["loss"] = [["0/1","1/1","1/1","1/1"]]+[["8/1","0/1","0/1","0/1"]]*3
        target, _, _ = self.assert_oracles(inp, False)
        from exact_sequential_inference.entropy_anchor import certificate
        self.assertEqual(certificate(inp)["max_information_tests"], [0])
        self.assertEqual(target["selected_action"], 5)
        self.assertEqual(dict(target["action_values"])[4], "27/100")
        self.assertEqual(target["minimum_expected_loss"], "1/50")

    def test_zero_mass_and_zero_branch(self):
        inp = tiny_active()
        inp["p_one"] = [["0/1", "0/1"], ["1/1", "1/1"]]
        root, trace, _ = self.assert_oracles(inp)
        self.assertTrue(any(n["mass"] == "0/1" for n in trace["nodes"]))
        inp["history"] = [{"test": 0, "outcome": 1}]
        target, trace, opts = self.assert_oracles(inp)
        self.assertIsNone(target["posterior"])
        self.assertIsNone(target["selected_action"])
        self.assertEqual(opts, [])

    def test_exhaustive_small_canonical_orbits(self):
        for cells in product(("0/1", "1/2", "1/1"), repeat=4):
            table = [list(cells[:2]), list(cells[2:])]
            key = active.canonical_table(table)
            for hp in permutations(range(2)):
                for tp in permutations(range(2)):
                    for flips in product((0,1), repeat=2):
                        transformed = [[qstr(1-Fraction(table[t][h]) if flips[t] else Fraction(table[t][h])) for h in hp] for t in tp]
                        self.assertEqual(key, active.canonical_table(transformed))
        # No merging of distinct orbit representatives in this finite grid.
        orbit_sets = {}
        for cells in product(("0/1", "1/2", "1/1"), repeat=4):
            table = [list(cells[:2]), list(cells[2:])]
            key = active.canonical_table(table)
            orbit_sets.setdefault(key, set()).add(tuple(cells))
        for orbit in orbit_sets.values():
            representative = next(iter(orbit))
            table = [representative[:2], representative[2:]]
            explicit = {tuple(qstr(1-Fraction(table[t][h]) if flips[t] else Fraction(table[t][h])) for t in tp for h in hp) for hp in permutations(range(2)) for tp in permutations(range(2)) for flips in product((0,1), repeat=2)}
            self.assertEqual(orbit, explicit)

    def test_malformed_active(self):
        mutations = [lambda x: x.update(initial_horizon=True), lambda x: x.update(initial_horizon=4), lambda x: x.update(extra=0), lambda x: x.update(prior=["1/1","1/1"]), lambda x: x["prior"].__setitem__(0, True), lambda x: x.update(history=[{"test": True,"outcome":0}]), lambda x: x.update(history=[{"test":0,"outcome":False}]), lambda x: x.update(history=[{"test":0,"outcome":0},{"test":0,"outcome":1}]), lambda x: x.update(hypotheses=["H0","H0"]), lambda x: x.update(p_one=[["2/1","0/1"],["0/1","0/1"]]), lambda x: x.update(costs=["-1/1","0/1"]), lambda x: x.update(loss=[["0/1"]]), lambda x: x.update(history=""), lambda x: x.update(tests=None)]
        for index, mutate in enumerate(mutations):
            inp = tiny_active()
            mutate(inp)
            with self.subTest(index=index), self.assertRaises(InvalidInput):
                active.solve(inp)


class TemporalTests(unittest.TestCase):
    def assert_oracles(self, inp, reports, queries):
        target, trace = temporal.solve(inp, reports, queries)
        self.assertEqual(dumps(target), dumps(temporal_reference.solve(inp, reports, queries)))
        self.assertEqual(dumps(target), dumps(temporal_reference.solve(inp, reports, queries, exhaustive=True)))
        return target, trace

    def test_exhaustive_three_step_reports_delays_parameters(self):
        # 3 hazards * 2 errors * 3 priors * 3^3 report sets * 2 delay schedules.
        cases = 0
        for hazard, error, prior, observations, delayed in product(("0/1","1/4","1/1"), ("0/1","1/5"), (["0/1","1/1"],["1/2","1/2"],["1/1","0/1"]), product((None,0,1), repeat=3), (False,True)):
            inp = tiny_temporal(hazard=hazard, error=error)
            inp["prior"] = prior
            reports = [report(t=t,b=b,arrival=min(2,t+1) if delayed else t,origin=f"o{t}") for t,b in enumerate(observations) if b is not None]
            queries = [dict(query(t=t,cutoff=c), query_id=f"q{t}_{c}") for c in range(3) for t in range(c+1)]
            self.assert_oracles(inp,reports,queries)
            cases += 1
        self.assertEqual(cases, 972)

    def test_eight_step_enumeration_and_asymmetric_declared_model(self):
        inp = tiny_temporal(8)
        inp["prior"] = ["1/3","2/3"]
        inp["transition"] = [["4/5","1/5"],["1/3","2/3"]]
        inp["sources"][0]["emission"] = [["9/10","1/10"],["1/4","3/4"]]
        reports = [report(t=t,b=t%2,origin=f"o{t}") for t in range(8)]
        for t in range(8):
            self.assert_oracles(inp,reports,[query(t=t,cutoff=7)])

    def test_delayed_arrival_smoothing_filter_and_as_known(self):
        inp = tiny_temporal(3, hazard="0/1")
        reports = [report(t=0,b=1,arrival=2)]
        early, _ = self.assert_oracles(inp,reports,[query(t=0,cutoff=0,kind="historical_as_known")])
        late, _ = self.assert_oracles(inp,reports,[query(t=0,cutoff=2)])
        current, _ = self.assert_oracles(inp,reports,[query(t=2,cutoff=2,kind="current_filter")])
        smooth_same, _ = self.assert_oracles(inp,reports,[query(t=2,cutoff=2)])
        self.assertEqual(early[0]["posterior"],["1/2","1/2"])
        self.assertEqual(late[0]["posterior"],["1/5","4/5"])
        self.assertEqual(current,smooth_same)
        view = temporal_payload(inp,reports,query(t=0,cutoff=0,kind="historical_as_known"))
        self.assertEqual(view["source_records"], [])
        self.assertNotIn("query_id",view["query"])
        self.assertEqual(set(adaptation_payload(inp,reports,0)), {"formal_input","cutoff","source_records"})
        tainted_reports=[dict(reports[0], oracle_posterior="secret")]
        tainted_input=dict(inp, latent_states="secret")
        clean=adaptation_payload(tainted_input,tainted_reports,2)
        self.assertNotIn("latent_states",clean["formal_input"])
        self.assertNotIn("oracle_posterior",clean["source_records"][0])

    def test_duplicates_permutations_corroboration(self):
        inp = tiny_temporal(3, hazard="0/1")
        base = [report(t=0,b=1,arrival=1)]
        copies = [report(t=0,b=1,arrival=2,rid="later"), *base, report(t=0,b=1,arrival=1,rid="same_time")]
        queries = [query(t=0,cutoff=2)]
        a, _ = self.assert_oracles(inp,base,queries)
        for perm in permutations(copies):
            b, _ = self.assert_oracles(inp,list(perm),queries)
            self.assertEqual(a,b)
            self.assertEqual(temporal_group(inp,base),temporal_group(inp,list(perm)))
        independent = base + [report(t=0,b=1,arrival=1,origin="second")]
        c, _ = self.assert_oracles(inp,independent,queries)
        self.assertEqual(c[0]["posterior"],["1/17","16/17"])
        self.assertNotEqual(a,c)
        self.assertNotEqual(temporal_group(inp,base),temporal_group(inp,independent))

    def test_global_impossible_evidence_other_entity(self):
        inp = tiny_temporal(2,entities=2,error="0/1")
        reports = [report(t=0,b=0,entity="E1",origin="a"),report(t=0,b=1,entity="E1",origin="b")]
        target,_ = self.assert_oracles(inp,reports,[query(t=0,cutoff=1)])
        self.assertEqual(target[0]["entity_evidence_probability"],"1/1")
        self.assertEqual(target[0]["status"],"impossible_evidence")
        self.assertIsNone(target[0]["posterior"])

    def test_renaming_grouping(self):
        inp = tiny_temporal(3,entities=2)
        reports = [report(t=0,b=1,entity="E1",origin="a"),report(t=1,b=0,entity="E0",origin="b")]
        group = temporal_group(inp,reports)
        renamed = copy.deepcopy(inp)
        renamed["entities"] = ["FreshA","FreshB"]
        renamed["sources"][0]["source_id"] = "FreshSource"
        remap = {"E0":"FreshB","E1":"FreshA"}
        new_reports = [dict(r,entity=remap[r["entity"]],source_id="FreshSource",origin_report_id="new"+r["origin_report_id"],report_id="new"+r["report_id"]) for r in reports]
        self.assertEqual(group,temporal_group(renamed,new_reports))

    def test_malformed_temporal(self):
        for field,value in [("event_time",True),("arrival_time",False),("value",True),("event_time",-1),("arrival_time",3),("entity",1),("source_id",[])]:
            inp=tiny_temporal()
            reports=[report()]
            reports[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(InvalidInput):
                temporal.solve(inp,reports,[query()])
        for q in [dict(query(),cutoff=True),dict(query(),target_time=False),dict(query(),kind="historical_as_known"),dict(query(),kind="current_filter"),dict(query(),target_time=3),dict(query(),extra="bad")]:
            with self.assertRaises(InvalidInput):
                temporal.solve(tiny_temporal(),[],[q])
        for reports in [[report(),report()], [report(),report(b=0,rid="conflict")], [report(t=1,arrival=0)], [report(),dict(report(rid="other"),source_id="not_known")]]:
            with self.assertRaises(InvalidInput):
                temporal.solve(tiny_temporal(),reports,[query()])
        inp=tiny_temporal()
        inp["steps"]=True
        with self.assertRaises(InvalidInput):
            temporal.solve(inp,[],[query()])
        with self.assertRaises(InvalidInput):
            score_temporal({"status":"ok","posterior":[True,"0/1"]},{"status":"ok","posterior":["1/1","0/1"]})


if __name__ == "__main__":
    unittest.main()
