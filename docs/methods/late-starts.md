# Late starts and no loss at an empty context

In use since 2026-11-23, in the published model `data/models/aximoe-261123`.

## What it does

Three in ten of the tiles generated afresh start late (`--stream-late 0.3`,
`synthetic.late_start`): every step before a drawn start is missing, and the start lies inside a
patch, so that the first patch with values holds at least two of them. With
`--empty-context-loss off` a position whose context holds no observed point has no loss, since
there is nothing to forecast from.

## Why

A series that begins inside the window is what a short history looks like: a new container, a
metric that just appeared, a benchmark series shorter than the window. On a public benchmark's
short series the model before this one met a context it had not been trained on: a window of 2,012
missing steps and four values got a median of 13.4 asinh units, 3.3e5 standard deviations in the
series' own units; five such windows of 2,000 set the configuration's score. Trained for 50,000
further steps onto the pulse-train model, late starts were level with their control on BOOMLET
(MASE 0.9997, 95 % paired interval 0.9992 to 1.0002, channel tag unknown) and 0.5 % behind it on
GIFT-Eval Web's MASE (1.0047, 1.0003 to 1.0116); they went into the full-length run with the
pulse trains.

## Limits

The cost on GIFT-Eval Web sits in two configurations. On BOOMLET a model trained with late starts
needs the forecaster's frame at the scale floor ([frame](frame.md)).
