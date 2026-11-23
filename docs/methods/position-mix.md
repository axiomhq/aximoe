# The mix of four positions

In use since 2026-11-15, in the release `aximoe-261115`
(`model.json` config: `canon` 4).

Since 2026-11-23 the published model `aximoe-261123` keeps the mix, over patches of 16 steps and beside rotary positions ([rotary positions](rotary-positions.md)); 4,036,464 parameters in all.

## What it does

Each layer adds to every position, per dimension, a learned weighted sum of that position and the
three before it: once before the attention and once before the feed-forward block
(`student.mix_back`, `--canon 4`). The weights start at zero, so the layer starts as the one
without the mix. It is causal like the attention, and the cached one-position pass keeps what it
reads (`student.mix_back_at`). It adds 8,192 parameters, 4,364,384 in all.

## Why

Attention reads any earlier patch but has no built-in sense of "the patch just before"; a short
local mix gives each layer that for a few parameters. The idea comes from language-model
research, where such layers are called Canon layers. At 50,000 steps, with an unroll of 8, the rollout's BOOMLET MASE read
0.9902 of the reference with the mix and 1.0023 without.
