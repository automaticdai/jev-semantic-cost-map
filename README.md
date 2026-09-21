# jev-motion-planner

Motion planning where the geometry is code and the judgment is [Jev](https://docs.typesafe.ai).

A warehouse floor plan gives you occupancy. It does not tell you that the spill
in aisle 3 was mopped twenty minutes ago, or that the stock counter in aisle 2 is
stepping in and out of the lane. jev-motion-planner hands each named zone to Jev
as four typed questions, turns the answers into a cost layer, and routes four
AGVs over it with a space-time A\*.

| t | What the shift log says | Keyword rules | Jev | Truth |
| --- | --- | --- | --- | --- |
| 3 | aisle 3: "Spill mopped, floor dried and signed off by the shift lead." | blocked (`spill` still in the accumulated notes) | x2.77, not blocked (`people` 0.86/3, `offlimits` P=0.08) | clear |
| 4 | aisle 2: "Stock count under way... the counter is stepping in and out of the lane." | x1.0 (no rule fired) | x8.97, not blocked (`people` 2.93/3 conf 0.93, `offlimits` P=0.25) | blocked |
| 7 | aisle 5: "Hydraulic fluid leaking... not yet cleaned." | blocked | blocked (`damage` 2.85/3, `offlimits` P=0.66) | blocked |

Row 7 is the point as much as rows 3 and 4: the rule list is a fair opponent and
it gets the obvious cases right. What it cannot do is read a correction, a time,
or a spatial qualifier.

Row 4 comes with a caveat worth stating plainly: Jev's `people` judgment reads
the counter correctly and prices the zone at x8.97 — the single highest
multiplier anywhere in the whole twelve-tick run — but its `offlimits` gate
stayed under the 0.5 threshold (0.25), so the report's strict pass/fail scoring
still counts this row as "both wrong" against the "blocked" ground truth. In
practice no AGV needed aisle 2 while the count was running, so the miscall
never produced an actual violation, but it means Jev priced the risk without
ever calling the zone closed.

The full picture also has a row where the baseline wins outright. At tick 9
(14:30), aisle 5's note is unchanged since tick 7 — the leak is not cleaned
until tick 10 — yet Jev's `offlimits` probability drifted from 0.66 (t7) to
0.60 (t8) to 0.44 (t9) with no new information, crossing the block threshold a
tick early. The keyword baseline, which just re-matches `leak` against the
same static text every tick, held the block correctly. `disagree` records this
as `baseline right`. Two ticks later (9 and 10), the same run shows the
opposite: a stalled AGV closes junction-4 by a standing-rule judgment call
("blocked", `offlimits` P > 0.5) that no keyword in `config/rules.yaml` matches
("stalled" fires nothing), so the baseline drives straight through it — the
run's one actual `blocked violation`, versus zero for Jev.

## What Jev decides, and what it does not

Jev answers four questions per zone: how exposed a person would be, how likely a
pass is to damage something, how much delay to expect, and whether the zone is
closed outright. Everything else is ordinary code — occupancy, collision
checking, search, reservations, cost arithmetic, rendering.

## Install

```bash
uv sync
export TYPESAFE_API_KEY=...   # from https://console.typesafe.ai/
```

## Use

```bash
jev-planner run --scenario config/scenarios/night-shift.yaml   # plan the shift
jev-planner run --scenario ... --dry-run     # print the state and questions, no call
jev-planner replay runs/example-night-shift  # re-render offline, no key needed
jev-planner explain runs/example-night-shift --zone aisle-3 --tick 3
jev-planner explain runs/example-night-shift --zone aisle-2 --tick 4
jev-planner disagree runs/example-night-shift
```

A twelve-tick shift is 94,431 input tokens (about 7,870 per tick, one request
per tick), roughly $0.0040 at Jev's current price of $0.042 per million input
tokens.

## Scoring

Every tick of a scenario carries hidden ground-truth labels, authored by hand and
never sent to the model. Both planners are scored against them, so the report says
which one was right rather than only that the paths differed.

Totals over the recorded night-shift run (`runs/example-night-shift/report.md`):

| Measure | jev | baseline |
| --- | ---: | ---: |
| blocked violations | 0 | 1 |
| avoid traversals | 8 | 8 |
| false blocks | 0 | 11 |
| path cells | 2770 | 2671 |
| deferrals | 2 | 2 |

Jev never drove an AGV into a zone the ground truth called `blocked`, and never
falsely blocked a zone that was actually `clear` — the baseline's 11 false
blocks are almost entirely aisle-3 after the spill was mopped, since the word
"spill" never leaves the accumulated notes. This is one twelve-tick run of one
hand-authored scenario, not a benchmark: the questions, the notes, and the
truth labels were all written by the people building the tool, and the tick-9
aisle-5 drift described above shows the model's own answers moving between
ticks on unchanged evidence. Treat the numbers here as a recorded anecdote
with ground truth, not a claim of general accuracy.

The baseline travels 99 fewer cells over the shift, but that is not a planning
win either way. Per-tick, the gap (jev cells minus baseline cells) is
10, 9, 4, 4, 2, 2, 0, 0, 0, 0, 38, 30 across ticks 0–11. The single biggest
piece, 38 cells at tick 10, is the baseline driving straight through the
stalled-AGV junction it never priced (junction-4, its one real
`blocked violation`, and the only tick where Jev's path touches zero
junction-4 cells against the baseline's 25). The 31 cells spread across ticks
0–5 predate that incident entirely and are just Jev routing more cautiously
through ordinary elevated-cost zones. The remaining 30 cells, at tick 11,
cut the other way: junction-4 is back to `clear` there, the incident is over,
and Jev's paths still pass through *more* junction-4 cells (48) than the
baseline's (25) — caution that this ground truth says was no longer needed.
Shorter paths are what you get for ignoring hazards, and longer ones are what
you get for staying wary past the point they resolve; this run has both.

## What it does not do

No policy re-weighting from the CLI, no confidence-gated escalation, no web
viewer, no continuous-space planning, no optimal multi-agent search. Prioritized
planning can starve a low-priority AGV; when it does, the report says so.
