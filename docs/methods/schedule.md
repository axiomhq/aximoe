# Cosine decay and the length of the run

In use since 2026-11-15, in the release `aximoe-261115`.

Since 2026-11-23 the published model `aximoe-261123` runs the same decay over 1,175,000 steps, 16 hours 35 minutes on an RTX 2080 Ti: with 13 tiles of a batch generated afresh instead of 6 ([pulse trains](pulse-trains.md)), the longer run draws the stored tiles as often as before. Its final weights are published; seed 1.

## What it does

After 200 warm-up steps the learning rate falls from 1.2e-3 to zero along half a cosine over the
run (`--lr-decay cosine`). The run of `aximoe-261115` is 1,043,796 steps, three times the
earlier schedule, 12 hours 40 minutes on an RTX 2080 Ti. Its final weights are published: under
the decay the last snapshots are too close for their mean to differ.

## Why

At 50,000 steps the decay took 0.9 % off BOOMLET's MASE (ratio 0.9914, 95 % paired interval
0.9795 to 1.0029), most of it on long horizons. Three times the schedule took the recipe's seed 0
from 0.6330 to 0.6303; means of the last two to four snapshots scored 0.6304 to 0.6309.

## Limits

A run under the decay cannot be extended: at its end the rate is zero, and the same weights are
not the midpoint of a longer run. `train.py` writes `resume.pt` every 5,000 steps so that a
crashed run continues under its own schedule (`--resume`).
