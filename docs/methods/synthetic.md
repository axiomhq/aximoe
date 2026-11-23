# Generated series

In use since 2026-11-15, in the release `aximoe-261115`.

Since 2026-11-23 the published model `aximoe-261123` draws 13 of every 64 tiles afresh, half of them pulse trains ([pulse trains](pulse-trains.md)) and three in ten with a late start ([late starts](late-starts.md)); the stored generated tiles are the same.

## What it does

`python/aximoe/synthetic.py` draws series of 2,048 steps at a nominal bin width: a periodic core
plus noise, trend and events, optionally exponentiated, floored, quantised and given gaps. Most
periods are natural durations (a minute to a week) that fit the bin width. The generator was
written from what telemetry looks like, not fitted to a benchmark.

- **Stored, with teacher labels.** 550,000 generated tiles are
  stored and forecast by Toto 2.0 313m like every tile; 495,202 of them are training tiles. Three
  draws in ten are one of them (`--synthetic-share 0.3`).
- **Fresh, truth only.** 6 of every 64 tiles in a batch are generated anew by worker processes
  and never seen again (`--stream-share 0.1 --stream-family any`); they have no teacher forecast
  and learn from the truth alone.

## Why

On public data the student stayed behind its teacher: what was missing were the inputs the
teacher was asked about. With the coarse widths
([coarse widths](coarse-widths.md)) the stored series took 9.6 % off BOOMLET's MASE at 50,000
steps (ratio 0.9045, 95 % paired interval 0.8747 to 0.9337); fresh series on the truth alone
reached 6.5 % in a run that was not paired. A stored share of 60 % was 1.5 % better than 30 % at
50,000 steps and worse at full length.

## Limits

By rollout on stored generated tiles held out of training the loss of `aximoe-261115` is 0.76 of
the previous model's; what the generator does not draw, the model has not seen. Each stored tile
is seen 40 times in the run of `aximoe-261115`.
