# Deep unroll, rollout only

In use since 2026-11-15, in the release `aximoe-261115`.

Since 2026-11-23 the published model `aximoe-261123` unrolls up to 17 patches of 16 steps on half of the steps and up to 61 on the other half (`--unroll 17 --unroll-depths 17:0.5,61:0.5`), 272 and 976 steps where the page's recipe reaches 256 and 960, at 19.7 steps a second on an RTX 2080 Ti in half precision ([patch width](patch-width.md)).

## What it does

The unroll ([band input](band-input.md)) replaces a tile's last context patches by the model's
own forecasts, median and band, made without gradients, before the training pass. The recipe of
`aximoe-261115` draws its depth per step: up to 8 patches on half of the steps, up to 30 on the
other half (`--unroll 8 --unroll-depths 8:0.5,30:0.5`); a tile with a stored hidden-patch
forecast is unrolled as that forecast was made ([far labels](far-labels.md)). No tile trains a
single-pass decoder (`--mask-tail 0`), so the model forecasts by rollout alone.

The one-position passes of the unroll read a cache of each layer's keys and values and replay as
a CUDA graph, which is what makes the depth affordable: 22.9 steps a second on an RTX 2080 Ti.

## Why

A forecast of several hundred steps is made patch by patch from the model's own output, so the
model has to be trained on its own output that far. At 50,000 steps an unroll of 8 on every tile
took 0.67 % off BOOMLET's MASE by rollout (ratio 0.9933, 95 % paired interval 0.9869 to 0.9997);
depths of 8, 16 and a mix were within 0.25 % of each other on BOOMLET and better the deeper on
the far patches of our own tiles.

## Limits

Training reaches 30 patches; a tile holds 64, so a deeper unroll leaves less context. BOOMLET's
long horizons run further than the model was trained to feed back.
