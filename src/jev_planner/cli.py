"""The jev-planner command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

    replay = sub.add_parser("replay", help="re-render a stored run; no network")
    replay.add_argument("run")

    explain = sub.add_parser("explain", help="every probability behind one zone's cost")
    explain.add_argument("run")
    explain.add_argument("--zone", required=True)
    explain.add_argument("--tick", type=int, default=0)

    disagree = sub.add_parser("disagree", help="where Jev and the rule baseline diverge")
    disagree.add_argument("run")
    return parser


def _cmd_run(args) -> int:
    from .render import render_run
    from .report import write_report

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
    render_run(scenario, run)
    write_report(scenario, run)
    print(f"{run.root}  {manifest['input_tokens']} tokens  ~${manifest['estimated_usd']}")
    return 0


def _cmd_replay(args) -> int:
    from .render import render_run
    from .report import write_report

    run = RunDir.open(args.run)
    scenario = load_scenario(run.read_manifest()["scenario_path"])
    print(render_run(scenario, run))
    write_report(scenario, run)
    return 0


def _open(path):
    run = RunDir.open(path)
    scenario_path = run.read_manifest()["scenario_path"]
    candidate = Path(scenario_path)
    if not candidate.exists():
        # The manifest stores the scenario path as typed at `run` time, which
        # is almost always relative to the repo root. That resolves fine when
        # an offline command is also run from the repo root, but not from
        # elsewhere. `run.root.parent.parent` is the repo root for the common
        # case of a run directory under `runs/<when>-<name>/` -- try the
        # scenario path relative to that before giving up. The manifest still
        # stores the relative path either way, so it stays machine-independent.
        alternate = run.root.parent.parent / scenario_path
        if alternate.exists():
            candidate = alternate
    return run, load_scenario(candidate)


def _cmd_explain(args) -> int:
    from .report import explain_text

    run, scenario = _open(args.run)
    try:
        print(explain_text(scenario, run, args.zone, args.tick))
    except KeyError:
        print(f"explain: no zone named {args.zone!r} in this scenario", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"explain: no tick {args.tick} recorded in {run.root}", file=sys.stderr)
        return 2
    return 0


def _cmd_disagree(args) -> int:
    from .report import disagree_text

    run, scenario = _open(args.run)
    try:
        print(disagree_text(scenario, run))
    except (KeyError, FileNotFoundError) as error:
        print(f"disagree: could not read {run.root}: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return {
        "run": _cmd_run,
        "replay": _cmd_replay,
        "explain": _cmd_explain,
        "disagree": _cmd_disagree,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
