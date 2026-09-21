# night-shift

Model `jev-1.13.0`, 94431 input tokens, about $0.0040.

## Scored against the hidden labels

| Measure | jev | baseline |
| --- | ---: | ---: |
| blocked violations | 0 | 1  |
| avoid traversals | 8 | 8  |
| false blocks | 0 | 11  |
| planned route cells | 2770 | 2671  |
| deferrals | 2 | 2  |

## Per tick

| tick | clock | jev blocked | base blocked | jev route cells | base route cells | deferred |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 13:00 | 0 | 0 | 392 | 382 | - |
| 1 | 13:10 | 0 | 0 | 346 | 337 | - |
| 2 | 13:20 | 0 | 0 | 300 | 296 | - |
| 3 | 13:30 | 0 | 0 | 254 | 250 | - |
| 4 | 13:40 | 0 | 0 | 206 | 204 | - |
| 5 | 13:50 | 0 | 0 | 158 | 156 | - |
| 6 | 14:00 | 0 | 0 | 110 | 110 | - |
| 7 | 14:10 | 0 | 0 | 62 | 62 | - |
| 8 | 14:20 | 0 | 0 | 115 | 115 | agv-3 |
| 9 | 14:30 | 0 | 0 | 79 | 79 | agv-3 |
| 10 | 14:40 | 0 | 1 | 396 | 358 | - |
| 11 | 14:50 | 0 | 0 | 352 | 322 | - |

## Where they disagreed

| tick | zone | jev | baseline | truth | verdict |
| ---: | --- | --- | --- | --- | --- |
| 0 | junction-1 | x3.5 | x1.0 | clear | degree only |
| 0 | junction-2 | x2.9 | x1.0 | clear | degree only |
| 0 | junction-3 | x4.8 | x1.0 | clear | degree only |
| 0 | junction-4 | x3.0 | x1.0 | clear | degree only |
| 0 | aisle-1 | x2.0 | x1.0 | clear | degree only |
| 0 | aisle-2 | x5.3 | x1.0 | clear | degree only |
| 0 | aisle-3 | x2.4 | x1.0 | clear | degree only |
| 0 | aisle-4 | x2.0 | x1.0 | clear | degree only |
| 0 | aisle-5 | x1.9 | x1.0 | clear | degree only |
| 0 | aisle-6 | x2.6 | x1.0 | clear | degree only |
| 0 | bay-A | x4.9 | x1.0 | clear | degree only |
| 0 | bay-B | x4.4 | x1.0 | clear | degree only |
| 0 | staging-7 | x5.4 | x1.0 | clear | degree only |
| 1 | junction-1 | x3.6 | x1.0 | clear | degree only |
| 1 | junction-2 | x3.5 | x1.0 | clear | degree only |
| 1 | junction-3 | x4.4 | x1.0 | clear | degree only |
| 1 | junction-4 | x3.3 | x1.0 | clear | degree only |
| 1 | aisle-1 | x2.4 | x1.0 | clear | degree only |
| 1 | aisle-2 | x4.3 | x1.0 | clear | degree only |
| 1 | aisle-3 | blocked | blocked | blocked | both right |
| 1 | aisle-4 | x2.0 | x1.0 | clear | degree only |
| 1 | aisle-5 | x1.8 | x1.0 | clear | degree only |
| 1 | aisle-6 | x3.2 | x1.0 | clear | degree only |
| 1 | bay-A | x3.0 | x1.0 | clear | degree only |
| 1 | bay-B | x2.8 | x1.0 | clear | degree only |
| 1 | staging-7 | x3.8 | x1.0 | clear | degree only |
| 2 | junction-1 | x3.4 | x1.0 | clear | degree only |
| 2 | junction-2 | x3.2 | x1.0 | clear | degree only |
| 2 | junction-3 | x3.1 | x1.0 | clear | degree only |
| 2 | junction-4 | x2.4 | x1.0 | clear | degree only |
| 2 | aisle-1 | x1.6 | x1.0 | clear | degree only |
| 2 | aisle-2 | x2.8 | x1.0 | clear | degree only |
| 2 | aisle-3 | blocked | blocked | blocked | both right |
| 2 | aisle-5 | x1.8 | x1.0 | clear | degree only |
| 2 | aisle-6 | x2.3 | x1.0 | clear | degree only |
| 2 | bay-A | x2.1 | x1.0 | clear | degree only |
| 2 | bay-B | x6.1 | x3.0 | avoid | degree only |
| 2 | staging-7 | x3.1 | x1.0 | clear | degree only |
| 3 | junction-1 | x3.4 | x1.0 | clear | degree only |
| 3 | junction-2 | x3.2 | x1.0 | clear | degree only |
| 3 | junction-3 | x2.9 | x1.0 | clear | degree only |
| 3 | junction-4 | x2.5 | x1.0 | clear | degree only |
| 3 | aisle-1 | x1.7 | x1.0 | clear | degree only |
| 3 | aisle-2 | x2.6 | x1.0 | clear | degree only |
| 3 | aisle-3 | x2.8 | blocked | clear | jev right |
| 3 | aisle-4 | x1.5 | x1.0 | clear | degree only |
| 3 | aisle-5 | x1.6 | x1.0 | clear | degree only |
| 3 | aisle-6 | x2.4 | x1.0 | clear | degree only |
| 3 | bay-A | x2.0 | x1.0 | clear | degree only |
| 3 | bay-B | x6.1 | x3.0 | avoid | degree only |
| 3 | staging-7 | x3.2 | x1.0 | clear | degree only |
| 4 | junction-1 | x3.5 | x1.0 | clear | degree only |
| 4 | junction-2 | x3.3 | x1.0 | clear | degree only |
| 4 | junction-3 | x2.8 | x1.0 | clear | degree only |
| 4 | junction-4 | x2.4 | x1.0 | clear | degree only |
| 4 | aisle-1 | x2.1 | x1.0 | clear | degree only |
| 4 | aisle-2 | x9.0 | x1.0 | blocked | both wrong |
| 4 | aisle-3 | x2.6 | blocked | clear | jev right |
| 4 | aisle-5 | x1.6 | x1.0 | clear | degree only |
| 4 | aisle-6 | x2.2 | x1.0 | clear | degree only |
| 4 | bay-A | x2.1 | x1.0 | clear | degree only |
| 4 | bay-B | x6.1 | x3.0 | avoid | degree only |
| 4 | staging-7 | x3.2 | x1.0 | clear | degree only |
| 5 | junction-1 | x3.2 | x1.0 | clear | degree only |
| 5 | junction-2 | x3.1 | x1.0 | clear | degree only |
| 5 | junction-3 | x2.7 | x1.0 | clear | degree only |
| 5 | junction-4 | x2.0 | x1.0 | clear | degree only |
| 5 | aisle-1 | x2.0 | x1.0 | clear | degree only |
| 5 | aisle-2 | x8.7 | x1.0 | blocked | both wrong |
| 5 | aisle-3 | x2.7 | blocked | clear | jev right |
| 5 | aisle-4 | x1.5 | x1.0 | clear | degree only |
| 5 | aisle-5 | x2.0 | x1.0 | clear | degree only |
| 5 | aisle-6 | x2.1 | x1.0 | clear | degree only |
| 5 | bay-A | x1.9 | x1.0 | clear | degree only |
| 5 | bay-B | x6.0 | x3.0 | avoid | degree only |
| 5 | staging-7 | x6.2 | x2.0 | avoid | degree only |
| 6 | junction-1 | x3.4 | x1.0 | clear | degree only |
| 6 | junction-2 | x2.6 | x1.0 | clear | degree only |
| 6 | junction-3 | x2.6 | x1.0 | clear | degree only |
| 6 | junction-4 | x2.0 | x1.0 | clear | degree only |
| 6 | aisle-1 | x1.9 | x1.0 | clear | degree only |
| 6 | aisle-2 | x8.3 | x1.0 | blocked | both wrong |
| 6 | aisle-3 | x2.7 | blocked | clear | jev right |
| 6 | aisle-6 | x2.1 | x1.0 | clear | degree only |
| 6 | bay-A | x2.1 | x1.0 | clear | degree only |
| 6 | staging-7 | x6.0 | x2.0 | avoid | degree only |
| 7 | junction-1 | x3.0 | x1.0 | clear | degree only |
| 7 | junction-2 | x2.4 | x1.0 | clear | degree only |
| 7 | junction-3 | x2.5 | x1.0 | clear | degree only |
| 7 | junction-4 | x1.9 | x1.0 | clear | degree only |
| 7 | aisle-1 | x2.1 | x1.0 | clear | degree only |
| 7 | aisle-2 | x8.4 | x1.0 | blocked | both wrong |
| 7 | aisle-3 | x2.7 | blocked | clear | jev right |
| 7 | aisle-5 | blocked | blocked | blocked | both right |
| 7 | aisle-6 | x1.6 | x1.0 | clear | degree only |
| 7 | bay-A | x1.8 | x1.0 | clear | degree only |
| 7 | staging-7 | x5.9 | x2.0 | avoid | degree only |
| 8 | junction-1 | x3.1 | x1.0 | clear | degree only |
| 8 | junction-2 | x2.4 | x1.0 | clear | degree only |
| 8 | junction-3 | x2.6 | x1.0 | clear | degree only |
| 8 | junction-4 | x2.0 | x1.0 | clear | degree only |
| 8 | aisle-1 | x1.9 | x1.0 | clear | degree only |
| 8 | aisle-2 | x2.9 | x1.0 | clear | degree only |
| 8 | aisle-3 | x2.6 | blocked | clear | jev right |
| 8 | aisle-5 | blocked | blocked | blocked | both right |
| 8 | aisle-6 | x1.5 | x1.0 | clear | degree only |
| 8 | bay-A | x1.8 | x1.0 | clear | degree only |
| 8 | staging-7 | x5.9 | x2.0 | avoid | degree only |
| 9 | junction-1 | x3.1 | x1.0 | clear | degree only |
| 9 | junction-2 | x2.4 | x1.0 | clear | degree only |
| 9 | junction-3 | x2.4 | x1.0 | clear | degree only |
| 9 | junction-4 | blocked | x1.0 | blocked | jev right |
| 9 | aisle-1 | x2.0 | x1.0 | clear | degree only |
| 9 | aisle-2 | x3.0 | x1.0 | clear | degree only |
| 9 | aisle-3 | x2.3 | blocked | clear | jev right |
| 9 | aisle-5 | x5.8 | blocked | blocked | baseline right |
| 9 | aisle-6 | x1.8 | x1.0 | clear | degree only |
| 9 | bay-A | x1.9 | x1.0 | clear | degree only |
| 9 | staging-7 | x5.2 | x2.0 | avoid | degree only |
| 10 | junction-1 | x3.0 | x1.0 | clear | degree only |
| 10 | junction-2 | x2.6 | x1.0 | clear | degree only |
| 10 | junction-3 | x2.3 | x1.0 | clear | degree only |
| 10 | junction-4 | blocked | x1.0 | blocked | jev right |
| 10 | aisle-2 | x2.9 | x1.0 | clear | degree only |
| 10 | aisle-3 | x2.1 | blocked | clear | jev right |
| 10 | aisle-5 | x2.5 | blocked | clear | jev right |
| 10 | bay-A | x2.2 | x1.0 | clear | degree only |
| 10 | staging-7 | x5.3 | x2.0 | avoid | degree only |
| 11 | junction-1 | x3.5 | x1.0 | clear | degree only |
| 11 | junction-2 | x2.8 | x1.0 | clear | degree only |
| 11 | junction-3 | x2.5 | x1.0 | clear | degree only |
| 11 | junction-4 | x2.7 | x1.0 | clear | degree only |
| 11 | aisle-1 | x1.9 | x1.0 | clear | degree only |
| 11 | aisle-2 | x2.9 | x1.0 | clear | degree only |
| 11 | aisle-3 | x2.1 | blocked | clear | jev right |
| 11 | aisle-5 | x2.4 | blocked | clear | jev right |
| 11 | aisle-6 | x1.5 | x1.0 | clear | degree only |
| 11 | bay-A | x1.7 | x1.0 | clear | degree only |
| 11 | staging-7 | x5.3 | x2.0 | avoid | degree only |
