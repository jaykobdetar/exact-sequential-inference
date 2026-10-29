"""Exact answer checks, including valid ties. No learned scorer."""
from fractions import Fraction
from .common import array, integer, obj, require, vector


def score_active(answer, target, optimal_actions):
    obj(answer, {"status", "posterior", "action"}, "active answer")
    require(answer["status"] in ("ok", "impossible_evidence"), "answer status")
    if answer["status"] == "impossible_evidence":
        require(answer["posterior"] is None and answer["action"] is None, "impossible answer must use null posterior/action")
        return {"correct": target["status"] == "impossible_evidence", "posterior_squared_error": None, "regret": None, "canonical_action": None}
    length = len(target["posterior"]) if target["posterior"] is not None else len(array(answer["posterior"], 2, 4, "posterior"))
    p = vector(answer["posterior"], length, True, True)
    action = integer(answer["action"], 0, 7, "answer action")
    if target["status"] != "ok":
        return {"correct": False, "posterior_squared_error": None, "regret": None, "canonical_action": False}
    values = dict(target["action_values"])
    require(action in values, "answer action unavailable at this history/horizon")
    truth = list(map(Fraction, target["posterior"]))
    mse = sum((a - b) ** 2 for a, b in zip(p, truth))
    regret = Fraction(values[action]) - Fraction(target["minimum_expected_loss"])
    from .common import qstr
    return {"correct": mse == 0 and action in optimal_actions, "posterior_squared_error": qstr(mse), "regret": qstr(regret), "canonical_action": action == target["selected_action"]}


def score_temporal(answer, target):
    obj(answer, {"status", "posterior"}, "temporal answer")
    require(answer["status"] in ("ok", "impossible_evidence"), "answer status")
    if answer["status"] == "impossible_evidence":
        require(answer["posterior"] is None, "impossible posterior must be null")
        return {"correct": target["status"] == "impossible_evidence", "posterior_squared_error": None}
    p = vector(answer["posterior"], 2, True, True)
    if target["status"] != "ok":
        return {"correct": False, "posterior_squared_error": None}
    from .common import qstr
    error = sum((a - Fraction(b)) ** 2 for a, b in zip(p, target["posterior"]))
    return {"correct": error == 0, "posterior_squared_error": qstr(error)}
