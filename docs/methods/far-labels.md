# The teacher's forecasts of hidden patches

In use since 2026-11-15, in the release `aximoe-261115`.

Since 2026-11-23 the published model `aximoe-261123` learns these forecasts on its 16-step grid: the passes below and their twins read from step 16 of each tile (`ahead16-313m/`), for the same tiles. At full length they are worth 0.9828 of BOOMLET's MASE (95 % paired interval 0.9741 to 0.9907) and 0.9771 of its CRPS (0.9651 to 0.9877) over the same model without them ([patch width](patch-width.md)).

## What it does

The store's teacher forecasts are one patch ahead: at every position Toto 2.0 313m has seen all
patches before the one it forecasts. The far-label pass runs the teacher a second
time on training tiles with their last 1 to 30 patches hidden, so it forecasts each hidden patch
from the context before them, in the asinh space of the statistics at the context's end. The
trainer reads these passes with `--ahead-teacher <dir>,<dir>,...`.

A tile with such a forecast is unrolled as the forecast was made ([deep unroll](deep-unroll.md)):
the same patches are replaced by the student's own forecasts, and at those positions the student
learns the teacher's hidden-patch quantiles beside the truth.

The pass covers about 33 % of the 10-second training tiles, 12 % of the 30-second ones, 89 % of
the 1-minute ones, and every coarse and every stored generated training tile.

## Why

A deeper unroll alone cost BOOMLET at 50,000 steps; the far forecasts as labels of the fed-back
positions are the part that paid. At full length the recipe with the unroll of up to 30 patches,
the frozen statistics and these labels scored 0.6330 MASE, against 0.6413 for the run with an
unroll of 8 and none of the three. Labels on every fine tile changed nothing further.
