# Rotary positions

In use since 2026-11-23, in the published model `data/models/aximoe-261123`
(`model.json` config: `rope` true).

## What it does

With `--rope on` every attention layer rotates its queries and keys by their patch position
before they meet: 32 pairs of dimensions per head, each pair at its own rate from a base of
10,000. The product of a query and a key then depends on how far apart their patches are, not on
where in the window they stand. The model has no learned position table: 4,036,464 parameters
against 4,069,232 with one. The cached one-position passes of the unroll and of the rollout
rotate the same way. A model has to be trained with it; weights trained with the table cannot
use it.

## Why

A rotated query and key carry the distance between two patches directly, where a position table
has to learn every place in the window. Against its twin with the position table (same code,
seed, steps and precision) the published model reads, paired:

| measure (reference: the twin with the position table) |  ratio | 95 % paired interval |
|-------------------------------------------------------|-------:|---------------------:|
| the twin (reference), every row                       | 1.0000 |                      |
| BOOMLET MASE, channel tag picked                      | 0.9981 |     0.9937 to 1.0021 |
| BOOMLET CRPS, channel tag picked                      | 0.9984 |     0.9946 to 1.0022 |
| GIFT-Eval Web MASE                                    | 0.9935 |     0.9789 to 1.0072 |
| GIFT-Eval Web CRPS                                    | 0.9815 |     0.9607 to 0.9996 |
| our unseen containers, rollout, levels 0.1 to 0.9     | 0.9929 |     0.9901 to 0.9955 |
| our unseen channels, rollout, levels 0.1 to 0.9       | 0.9936 |     0.9904 to 0.9965 |

At 50,000 steps the gain on BOOMLET was larger, 0.5 to 1.5 % in three pairs. Along the full run
the pair's distance on BOOMLET's MASE read 0.6 % at 100,000 steps, none at 200,000, 0.7 % at
300,000 and 0.2 % at the end. On our own tiles it is 0.6 to 0.7 % with intervals off 1.

## Limits

One seed with it and one without. On unseen containers the levels 0.99 and above are 20 to 46 %
worse than the twin's, with wide intervals; on unseen channels every outer level is better.
