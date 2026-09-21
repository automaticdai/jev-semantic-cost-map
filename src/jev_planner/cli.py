"""The jev-planner command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .baseline import load_rules
from .costs import load_weights
from .events import initial_agvs, load_scenario, state_at
from .judge import Judge
from .questions import build_questions
from .runs import RunDir
from .shift import ShiftAborted, run_shift
from .state import build_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-planner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan a whole shift")
    run.add_argument("--scenario", required=True)
    run.add_argument("--out", default="runs")
    run.add_argument("--weights", default="config/weights.yaml")
    run.add_argument("--rules", default="config/rules.yaml")
    run.add_argument("--model", default=None)
    run.add_argument("--dry-run", action="store_true",
                     help="print tick 0's state and questions; make no API call")
    return parser


def _cmd_run(args) -> int:
    scenario = load_scenario(args.scenario)
    if args.dry_run:
        ws = state_at(scenario, 0, initial_agvs(scenario.world))
        print(json.dumps({
            "state": build_state(scenario.world, ws),
            "questions": {
                qid: q.model_dump(mode="json")
                for qid, q in build_questions(scenario.world).items()
            },
        }, indent=1))
        return 0

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set. Create a key at https://console.typesafe.ai/ "
              "and export it, or use --dry-run.", file=sys.stderr)
        return 2

    run = RunDir.create(args.out, scenario.name)
    try:
        manifest = run_shift(
            scenario,
            Judge(model=args.model),
            run,
            load_weights(args.weights),
            load_rules(args.rules),
        )
    except ShiftAborted as aborted:
        print(f"{aborted}. Artifacts written so far are in {run.root}.", file=sys.stderr)
        return 1
    print(f"{run.root}  {manifest['input_tokens']} tokens  ~${manifest['estimated_usd']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return {"run": _cmd_run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
