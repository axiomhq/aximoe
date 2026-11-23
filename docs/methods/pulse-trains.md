# Generated pulse trains

In use since 2026-11-23, in the published model `data/models/aximoe-261123`.

## What it does

Of every 64 tiles in a batch 13 are generated afresh and trained on the truth alone
(`--stream-share 0.2`), and half of those come from the pulse family
(`--stream-family any-pulse`, `synthetic.pulse`): a baseline with one to four events per cycle
of a period, each at its own place in the cycle with its own height and shape, a single step, a
box, a decaying tail or a rise and fall, as timers, flushes and scheduled jobs draw them. Three
series in five repeat exactly; the others vary the height per cycle, miss a cycle or carry noise.
The period is a whole number of steps up to 64, any length from 2 to 700 steps, or a natural
duration at the tile's bin width. The other half are the generator's earlier families
([generated series](synthetic.md)).

## Why

Events that repeat exactly are common in telemetry. On generated spike trains with a period
under 32 steps a 50,000-step model of the earlier recipe missed with its median by 0.443 of the
series' mean absolute deviation, and by 0.168 with the family in a fresh share of 0.2. At that
length BOOMLET's MASE read 0.9922 of the same recipe without the family (95 % paired interval
0.9845 to 0.9978, channel tag unknown). At the length of `aximoe-261115`, 1,043,796 steps,
against that model with the tag picked: BOOMLET MASE 0.9946
(0.9895 to 0.9986) and CRPS 0.9961 (0.9923 to 0.9993), GIFT-Eval Web MASE 0.9818 (0.9715 to
0.9918), level by rollout on our own unseen containers (1.0020, 0.9994 to 1.0047).

## Limits

The long periods stay far off: at a period of 128 steps the probe's error did not move with the
share. The fresh share doubles, so the run is longer by an eighth to draw the stored tiles as
often ([schedule](schedule.md)).
