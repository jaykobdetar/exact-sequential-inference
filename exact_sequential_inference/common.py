"""Strict JSON types, reduced rational codec, and reproducible serialization."""
import hashlib
import json
import math
import re
from fractions import Fraction


class InvalidInput(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def obj(value, keys, where):
    require(type(value) is dict and set(value) == set(keys), f"{where}: exact keys required: {sorted(keys)}")


def integer(value, low, high, where):
    require(type(value) is int and low <= value <= high, f"{where}: integer in [{low}, {high}] required (bool forbidden)")
    return value


def array(value, low, high, where):
    require(type(value) is list and low <= len(value) <= high, f"{where}: array length in [{low}, {high}] required")
    return value


def name(value, where):
    require(type(value) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,79}", value) is not None, f"{where}: bounded identifier required")
    return value


def names(value, low, high, where):
    array(value, low, high, where)
    for x in value:
        name(x, where)
    require(len(set(value)) == len(value), f"{where}: distinct names required")
    return value


def rational(value, probability=False):
    require(type(value) is str and len(value) <= 4096 and re.fullmatch(r"(?:0|[1-9][0-9]*)/[1-9][0-9]*", value) is not None, "nonnegative canonical rational string required")
    a, b = map(int, value.split("/"))
    require(math.gcd(a, b) == 1, "rational must be reduced")
    require(not probability or a <= b, "probability outside [0,1]")
    return Fraction(a, b)


def qstr(value):
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def vector(value, length, probability=False, normalized=False):
    array(value, length, length, "vector")
    out = tuple(rational(x, probability) for x in value)
    require(not normalized or sum(out) == 1, "probabilities must sum exactly to one")
    return out


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    def invalid(value):
        raise InvalidInput(f"nonfinite JSON constant: {value}")
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=unique, parse_constant=invalid)
