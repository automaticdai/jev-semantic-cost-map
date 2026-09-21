# jev-semantic-cost-map — design

Date: 2026-09-21
Status: approved, ready for implementation planning

## What this is

A demo that puts [Jev](https://docs.typesafe.ai) in charge of the *semantic* layer of
path planning for warehouse AGVs, and leaves every geometric decision to ordinary
code.

A warehouse floor is described in YAML: static geometry plus named zones with
natural-language descriptions. A scripted event log ticks through a shift, changing
only those descriptions — a spill reported in aisle 3, pickers working bay B until
14:00, glassware staged in staging-7. Each tick, one Jev request judges every zone
against the current scene, code fuses those judgments into a cost layer, and a
prioritized space-time A* routes four AGVs over it. The paths visibly reroute as the
words change.

A keyword rule model plans the same scenarios alongside Jev, and every tick carries
hand-authored ground-truth labels that are never sent to the model, so the demo can
report which one was right rather than only that they differed.

## Why a semantic cost layer

Geometry cannot encode "the pallet wrap spill was mopped at 13:40" or "the staff
crossing bay B are between racks, not in the lane". A rule list can only match words
someone thought to add. Jev reads the scene as written and returns calibrated,
typed judgments that code turns into numbers it controls.

Everything Jev is not good at stays in code: occupancy, collision checking, search,
reservations, cost arithmetic, rendering.

## Scope

In scope for v1:

- Zone-semantic cost layer driven by Jev.
- Event-driven replanning across a scripted shift.
- Four AGVs, prioritized space-time A*.
- Keyword rule baseline planned in parallel, with a `disagree` view.
- Hidden ground-truth labels and a scoring report.
- CLI with run / replay / explain / disagree / dry-run, and a run directory of
  artifacts including PNG frames and an animated GIF.

Explicitly out of scope for v1 (recorded so the implementation does not drift into
them):

- Policy re-weighting from the CLI. Weights live in a config file and are read once.
- Confidence-gated escalation. Confidence is stored and displayed, not acted on.
- Any web viewer or interactive UI.
- Continuous-space or kinodynamic planning. The world is a grid.
- Optimal multi-agent planning (CBS and friends). Prioritized planning is enough.
- Real robot or ROS integration.

## Package and dependencies

Repository `jev-semantic-cost-map`, Python package `jev_costmap` under `src/`, console
script `jev-costmap`. Python 3.12, managed with `uv`.

Runtime dependencies: `typesafe-sdk`, `numpy`, `matplotlib`, `pyyaml`, `pillow`
(matplotlib's `PillowWriter` writes the GIF). Development: `pytest`.

## Architecture

```
scenario.yaml ─┐
events.yaml  ──┴─► WorldState(t) ──► state JSON ──┐
                        │                          ├─► one Jev request ─► answers(t)
                        │            questions ────┘                          │
                        │                                                     ▼
                        │                                          costs.py fusion
                        │                                                     │
                        ▼                                                     ▼
                  occupancy grid ───────────────────────────────────► cost layer(t)
                        │                                                     │
                        └──────────────► prioritized space-time A* ◄──────────┘
                                                     │
                                          plans(t) ──┴──► frame(t), report
```

Only `judge.py` touches the network. Everything downstream consumes stored answers,
so a recorded run re-renders offline with no API key.

### Modules

`src/jev_costmap/`

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `world.py` | Load scenario YAML; zones, stations, static geometry; rasterize to an occupancy grid and a zone-id grid | — |
| `events.py` | Load the event log; produce an immutable `WorldState` per tick by applying events to zone descriptions and AGV state | `world` |
| `state.py` | Serialize a `WorldState` to the JSON dict sent as Jev `state` | `world`, `events` |
| `questions.py` | Build the question dict for a `WorldState`: four questions per zone | `world` |
| `judge.py` | Call Jev once per tick through an injectable client; on-disk cache; parse and validate answers | `state`, `questions` |
| `costs.py` | Fuse answers into per-zone multipliers and a blocked set; rasterize to a cost layer | `world`, `judge` |
| `baseline.py` | The keyword rule cost model, producing the same cost-layer type | `world`, `events` |
| `planner.py` | Space-time A*; prioritized multi-AGV planning with vertex and edge reservations | `world`, `costs` |
| `render.py` | Matplotlib frames and the animated GIF | `world`, `costs`, `planner` |
| `report.py` | `report.md`, `explain`, `disagree`, ground-truth scoring | everything |
| `runs.py` | Run directory layout, manifest, read and write | — |
| `cli.py` | Argument parsing and command wiring | everything |

A module that grows past roughly 300 lines is a signal to split it; `planner.py`
and `report.py` are the likely candidates.

## The world

An 80 x 50 grid at 0.5 m per cell: 40 m by 25 m of floor.

Fourteen zones, each a rectangle with an id, a human-readable name, a kind, and a
static description that events extend:

| Zone ids | Kind |
| --- | --- |
| `aisle-1` … `aisle-6` | travel lane between racking |
| `junction-1` … `junction-4` | aisle crossings |
| `bay-A`, `bay-B` | pick bays where people work |
| `staging-7` | goods staged on the floor |
| `charging` | AGV charging bank |

The aisles run parallel, so rerouting is always geometrically possible and a reroute
is a real trade of distance against semantic cost rather than the only option.

Four AGVs, each with a queue of station-to-station tasks. Every tick, each AGV
advances `cells_per_tick` cells along its current plan (declared in the scenario,
default 4), then all AGVs replan from their new positions. The primary scenario
runs 12 ticks.

## The state sent to Jev

One request carries one `state`. Every question is evaluated against it in parallel,
so a tick is one request covering the whole warehouse. Re-judging every zone each
tick is also the correct semantics: an incident in aisle 3 changes what aisle 4
means, because traffic diverts into it.

```json
{
  "shift": {"clock": "13:48", "name": "night-shift"},
  "zones": {
    "aisle-3": {
      "name": "Aisle 3",
      "kind": "travel lane",
      "description": "Travel lane between pallet racking, rated for AGV traffic.",
      "notes": [
        "13:31 Pallet wrap spill reported at the north end.",
        "13:40 Spill mopped and signed off by the shift lead."
      ]
    }
  },
  "agvs": {
    "agv-2": {"position": "junction-2", "committed_route": ["aisle-4", "junction-3", "bay-A"]}
  }
}
```

`committed_route` is the route committed at the **previous** tick. Using the current
tick's route would be circular — judgments feed planning, not the other way around —
and the previous route is what a real dispatcher actually knows. The manifest
records the tick each route came from so a stale route is visible rather than
assumed fresh.

## The judgments

Four questions per zone, with the zone named explicitly in the instructions via a
backticked path such as `` `zones.aisle-3` ``. Question ids are `{zone}.{dimension}`
and exist for code; the full meaning is in the question text.

### `{zone}.people` — Score, 4 levels

Instructions: how exposed would a person be if an AGV drove through this zone right
now.

Levels, each describing a concrete situation:

0. No one works in this zone and no one is expected in it during this shift.
1. People pass through occasionally but no one is working here now.
2. People are working in this zone now, clear of the travel lane.
3. People are working in or across the travel lane, where an AGV would pass.

### `{zone}.damage` — Score, 4 levels

Instructions: how likely is an AGV passing through this zone right now to damage
goods, the floor surface, or itself.

0. Nothing here is vulnerable; a clear, rated travel surface.
1. Goods are present but robust and clear of the lane.
2. Fragile goods, loose material, or an obstruction sits close to the lane.
3. The surface or the load is compromised in a way that makes a pass likely to
   cause damage or a loss of traction.

### `{zone}.delay` — Score, 4 levels

Instructions: how much would an AGV be slowed by congestion or obstruction in this
zone right now, ignoring safety and damage.

0. Clear; an AGV passes at full speed.
1. Light traffic or a narrow point; a short slowdown.
2. Busy; an AGV would likely have to stop and wait.
3. Effectively impassable without a long wait.

### `{zone}.offlimits` — Noul

Is this zone closed to AGV traffic right now by an explicit instruction, an active
incident, or a standing rule?

### Why four and not one

A single fused "how costly is this zone" Score would reroute correctly and cost
fewer tokens. It is rejected because `explain` would have nothing to show beyond one
number, and the `disagree` view would degenerate to "the two numbers differ". The
four dimensions are what make a divergence explicable: the baseline blocked aisle 3
on the word *spill*, Jev's `damage` score fell to 0.4 after the 13:40 note, and the
path went back through it.

## Cost fusion

In `costs.py`, per zone:

1. If `offlimits.noul >= 0.5`, the zone is blocked: every cell in it is impassable
   this tick. This is a hard gate, not a weight, because "closed" is not a quantity.
2. Otherwise each Score's expected value (`answer.score`, already
   probability-weighted across levels) is normalized to `[0, 1]` by dividing by
   `levels - 1`.
3. `multiplier = 1 + (w_people * people + w_damage * damage + w_delay * delay)`
4. Cell cost = base traversal cost x the multiplier of the zone containing the
   cell. Base traversal cost is 1.0 for every free cell; the grid carries no
   geometric cost of its own, so every cost difference comes from a judgment.

Weights live in `config/weights.yaml`, read once at run start and recorded in the
manifest. Starting values: `people: 6.0`, `damage: 3.0`, `delay: 1.5`. They are a
starting point to be tuned against the scenarios during implementation; the demo
does not depend on these particular numbers.

Every raw answer — each level probability, each confidence, the Noul probability —
is stored verbatim in `answers/tNN.json`. Nothing downstream may discard them;
`explain` reads them directly.

Confidence is stored and displayed but does not change any cost in v1.

## Multi-robot planning

Prioritized space-time A*:

- AGVs are planned in a fixed priority order declared in the scenario.
- Each AGV searches over `(cell, timestep)` on the fused cost layer.
- Each planned path becomes a time-indexed reservation for every AGV planned after
  it, forbidding both **vertex** conflicts (two AGVs in one cell at one timestep)
  and **edge** conflicts (two AGVs swapping cells between timesteps). Without the
  edge check, two AGVs pass through each other in a one-cell-wide aisle; the frame
  looks correct and the plan is not.
- Waiting in place is a legal action at a small fixed cost, independent of the
  zone multiplier.
- The search is capped at a horizon of 400 timesteps. An AGV with no path within
  the horizon holds position, is recorded as `deferred` for that tick, and retries
  next tick.

Prioritized planning is incomplete: a low-priority AGV can be starved by higher
priority traffic. This is accepted for a demo about semantics, and deferrals are
reported per tick rather than hidden, so starvation is visible when it happens.

## The baseline

`config/rules.yaml` holds an ordered list of regex rules over a zone's description
and notes, each yielding a multiplier or a block:

| Pattern | Effect |
| --- | --- |
| `spill\|wet\|leak` | blocked |
| `picker\|staff\|operator\|person\|people` | x3 |
| `fragile\|glass\|breakable` | x2 |
| `closed\|do not enter` | blocked |
| (no match) | x1 |

This is a fair strawman: it is what a keyword cost model in production actually
looks like, and it gets the obvious cases right. The demo's claim is not that rules
are stupid, it is that rules cannot read a correction, a time, or a spatial
qualifier.

The baseline costs nothing to evaluate, so **every run plans both models**. Baseline
plans are stored alongside Jev plans, and `disagree` is pure reporting over stored
data with no network access.

## Ground truth and scoring

Each tick of a scenario carries a hidden `truth` map, one label per zone:

- `clear` — routing through is fine
- `avoid` — passable, but a planner should prefer an alternative
- `blocked` — a planner must not route through

These labels are authored by hand alongside the events and are **never** included in
the state sent to Jev. `report.py` scores both models against them per tick:

- **blocked violations** — routed through a `blocked` zone (the serious error)
- **avoid traversals** — routed through an `avoid` zone
- **false blocks** — treated a `clear` zone as impassable
- **path length** — total cells travelled, so caution can be priced against distance
- **deferrals** — AGVs that could not plan

Divergences are reported in both directions, including ticks where the baseline
scores better. Without the hidden labels the demo could only say the paths differ,
which is an anecdote.

## CLI

```
jev-costmap run --scenario config/scenarios/night-shift.yaml [--out runs/<id>]
jev-costmap run --scenario ... --dry-run     # print tick 0 state + questions, no call
jev-costmap replay runs/<id>                  # re-render from stored answers, offline
jev-costmap explain runs/<id> --zone aisle-3 [--tick 4]
jev-costmap disagree runs/<id>
```

`replay`, `explain`, `disagree` and `--dry-run` require no API key.

## Run directory

```
runs/2026-09-21T14-02-night-shift/
  manifest.json      scenario, model id resolved from the response, git sha,
                     weights, per-tick status, token usage, cost
  states/t00.json    the exact state sent
  questions/t00.json the exact questions sent
  answers/t00.json   the raw response, every probability and confidence
  costs/t00.json     jev and baseline per-zone multipliers and blocked flags
                     (not the rasterized grid: `rasterize()` rebuilds that
                     deterministically from these, so json is smaller,
                     diffable, and provably consistent with what was
                     actually planned over -- an implementation improvement
                     over this doc's original `.npz`, kept as the record of
                     what shipped)
  plans/t00.json     per-AGV paths and deferrals, both models
  truth/t00.json     the hidden labels, copied in for scoring
  frames/t00.png
  shift.gif
  report.md
```

A judgment cache at `.jev-cache.json`, keyed on a hash of
`(state, questions, model id)`, makes a repeated run free and matches jev-cleaner's
convention.

## Error handling

- Transient API errors retry through the SDK's `RetryPolicy`. On final failure the
  run aborts, keeps every artifact already written, and marks the tick `failed` in
  the manifest.
- A response missing an expected question id is a hard error. A question and answer
  mismatch is a bug in question construction, not a condition to paper over.
- Jev failing never falls back to the baseline. In a demo about judgment quality,
  a silent fallback would hide exactly what the demo exists to show.
- A missing `TYPESAFE_API_KEY` fails `run` immediately with a message pointing at
  the console, and does not affect the offline commands.

## Testing

Test-driven throughout. The default suite makes no network calls: `judge.py` takes
an injectable client, and fixtures are real responses recorded from one live run
and stored under `tests/fixtures/answers/`.

| Test | Covers |
| --- | --- |
| `test_world.py` | zone rasterization, grid bounds, overlapping-zone rejection |
| `test_events.py` | event application, tick ordering, immutability of prior states |
| `test_state.py` | golden state JSON; truth labels never appear in it |
| `test_questions.py` | golden question dict; four per zone; ids unique |
| `test_judge.py` | answer parsing, missing-id error, cache hit and miss, retry |
| `test_costs.py` | normalization, weighted fusion, the 0.5 block threshold |
| `test_planner.py` | A* on hand-checked grids, an edge-swap case, an unsolvable case producing a deferral, wait actions |
| `test_baseline.py` | rule matching and precedence |
| `test_report.py` | scoring arithmetic against known truth, disagree in both directions |
| `test_cli.py` | command wiring, offline commands with no key set |
| `test_live.py` | one `@pytest.mark.live` smoke test, skipped without a key: a real request round-trips with every expected question id |

`test_state.py` asserting that the hidden truth labels never reach the state is the
test that protects the whole evaluation from becoming circular.

## Deliverables

- The package, CLI and tests above.
- `config/scenarios/night-shift.yaml` — the primary scenario, with events, hidden
  truth labels, and AGV tasks.
- A committed example run under `runs/`, so the repo demonstrates itself with no
  key and no network.
- `README.md` in the style of jev-cleaner: what it does, the one table that makes
  the case, install, use, and an honest account of what the baseline gets right.
