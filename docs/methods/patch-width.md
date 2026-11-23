# Patches of 16 steps

In use since 2026-11-23, in the published model `data/models/aximoe-261123`
(`model.json`: `patch` 16, `context` 2032).

## What it does

The model reads and forecasts 16 steps at a time: a window of 2,032 steps is 127 patches, and a
forecast of 32 steps takes two passes, the second made after reading the first one's median and
band. A process runs at one patch width, `AXIMOE_PATCH`; `forecast.py` and `mcp_server.py` take it
from the model they load, and the benchmark harness starts the forecaster at the model's width.

The stores stay as they are: tiles of 2,048 steps with Toto 2.0 313m's forecasts at boundaries
32 steps apart. For the boundaries in between, a second pass runs the teacher
on every tile from step 16 on (`shift16-313m/` beside the store, named to the trainer by
`AXIMOE_SHIFT_TEACHER`), and the trainer computes a tile's scaler statistics on the model's grid.
The trainer's evaluation scores the model on the store's grid, each 32-step position with the
forecasts of its two 16-step positions, the later one made after reading the earlier one's steps.

## Why

A forecast is read from the last patch of the context, so a finer patch lets the model answer
from a more recent point and correct itself twice as often along a rollout. At full length, with
the teacher's far forecasts on its grid ([far labels](far-labels.md)), the 16-step model scored
0.9929 of the 32-step model of the same recipe, steps and precision on BOOMLET's MASE (95 %
paired interval 0.9898 to 0.9963) and 0.9935 on its CRPS (0.9908 to 0.9962). The width alone did
not pay: the far labels are worth 0.9828 (0.9741 to 0.9907) and 0.9771 (0.9651 to 0.9877) over
the same 16-step model without them, more than the gain over the 32-step model.

## Limits

Twice the passes: training runs at 19.7 steps a second on an RTX 2080 Ti in half precision where
the 32-step recipe ran at 48.1, and the forecast in the MCP server's shape takes 2.2 to 2.3 times
as long (0.83 s against 0.36 s for 512 windows on a CPU). One step ahead on our own held-out
tiles the 16-step runs have 8 to 10 % more pinball on quiet channels than the 32-step run
(0.1773 against 0.1646 on held-out time, 0.1867 against 0.1701 on unseen containers) beside a
Winkler score 2.9 to 4.1 % lower on busy ones, from the trainers' final evaluations, unpaired.
On GIFT-Eval Web the width's gain on BOOMLET does not show: against the 32-step run MASE 0.9947
(0.9811 to 1.0094) and CRPS 1.0270 (1.0040 to 1.0560), the loss in the two 10-second datasets.
