"""CLI: generation, recomputation, inference, and safe per-query/source exports."""
import argparse
import json
from pathlib import Path
from .common import dumps, load_json, require


def main():
    parser = argparse.ArgumentParser(prog="python -m exact_sequential_inference")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--config", required=True)
    gen.add_argument("--output", required=True)
    val = commands.add_parser("validate")
    val.add_argument("dataset")
    val.add_argument("--report")
    sol = commands.add_parser("solve")
    sol.add_argument("mode", choices=["active_sensing", "noisy_temporal"])
    sol.add_argument("input")
    exp = commands.add_parser("export")
    exp.add_argument("dataset")
    exp.add_argument("--output", required=True)
    exp.add_argument("--view", choices=["model", "adaptation"], required=True)
    exp.add_argument("--cutoff", type=int, default=63)
    args = parser.parse_args()
    if args.command == "generate":
        from .generate import generate
        result = generate(load_json(args.config), args.output)
    elif args.command == "validate":
        from .validate import validate_dataset
        result = validate_dataset(args.dataset)
        if args.report:
            require(not Path(args.report).exists(), "report exists; preserve prior evidence")
            Path(args.report).write_text(dumps(result) + "\n")
    elif args.command == "solve":
        value = load_json(args.input)
        if args.mode == "active_sensing":
            from .active import solve
            target, trace, optimal = solve(value)
            result = {"oracle_targets": target, "exact_trace": trace, "optimal_actions": optimal}
        else:
            from .temporal import solve
            from .common import obj
            obj(value, {"formal_input", "source_records", "queries"}, "solve envelope")
            target, trace = solve(value["formal_input"], value["source_records"], value["queries"])
            result = {"oracle_targets": target, "exact_trace": trace}
    else:
        from .render import active_payload, temporal_payload, adaptation_payload
        from .common import integer
        integer(args.cutoff, 0, 63, "cutoff")
        output = Path(args.output)
        require(not output.exists(), "export exists; use a new path")
        count = 0
        with (Path(args.dataset) / "records.jsonl").open() as source, output.open("x", encoding="utf-8") as out:
            for line in source:
                r = json.loads(line)
                if r["mode"] == "active_sensing":
                    from .active import validate
                    validate(r["formal_input"])
                else:
                    from .temporal import validate
                    validate(r["formal_input"], r["source_records"], r["queries"])
                if args.view == "adaptation":
                    if r["mode"] != "noisy_temporal":
                        continue
                    inputs = [adaptation_payload(r["formal_input"], r["source_records"], args.cutoff)]
                elif r["mode"] == "active_sensing":
                    inputs = [active_payload(r["formal_input"])]
                else:
                    inputs = [temporal_payload(r["formal_input"], r["source_records"], q) for q in r["queries"]]
                for value in inputs:
                    out.write(dumps(value) + "\n")
                    count += 1
        result = {"exported_views": count, "path": str(output), "targets_included": False}
    print(dumps(result))


if __name__ == "__main__":
    main()
