# Student: inputs, backbone, head and the loss on the truth

Since 2026-11-15 the release `aximoe-261115` adds a mix of four positions to each layer: [position mix](position-mix.md). The rest of this page holds for it.

Since 2026-11-23 the published model `aximoe-261123` reads 127 patches of 16 steps ([patch width](patch-width.md)) with rotary positions in place of the position table ([rotary positions](rotary-positions.md)), 4,036,464 parameters. The rest of this page holds for it.

The shape of the first dense student of 2026-10-28 (four layers of width 256, fifteen monotone
levels), in PyTorch since 2026-10-31. The release `aximoe-261105` is this
network with the band input's wider patch embedding ([band input](band-input.md)), 4,351,840
parameters (train.log: 4.35 M); `model.json` config: `d_model` 256, `heads` 4, `layers` 4,
`d_ff` 688, `experts` 0, `band_input` true.

## What it does

**Inputs** (`batches.standardise`). A training tile is 64 patches of 32 steps; the student reads
them causally and position s forecasts patch s + 1, so positions 0 to 62 carry a target and the
last one sees 63 patches, 2,016 steps (`student.CONTEXT`). Every value becomes
`asinh((v - loc) / scale)` (asinh, the inverse hyperbolic sine) with the causal scaler's location
and scale at the last step of its own patch; target patch s + 1 uses those at the end of patch s.
In training they are the store's, the teacher's own ([teacher](teacher.md)); at inference
`forecast.causal_stats` computes them as Toto 2.0's `PatchedCausalStdScaler` does: running mean
and standard deviation (divided by n − 1) of the present values, accumulated in float64, read at
the end of each patch, the scale at least 1e-6. A missing step enters as zero, its mask as zero.

**Embedding** (`Student.hidden`). Each patch's 32 values and 32 mask values, `Linear(64, 1024)`,
SiLU (sigmoid linear unit, `x * sigmoid(x)`), `Linear(1024, 256)`; plus a learned vector for each
of the 64 positions (`pos`), the channel row and the resolution row ([tags](tags.md)).

**Backbone** (`Layer`). Four pre-norm layers of width 256: each block normalises its input and
adds its output back to the stream.
- RMSNorm (root-mean-square normalisation, eps 1e-4), then causal self-attention with four heads
  (`nn.MultiheadAttention` under an upper-triangular mask: a patch attends to itself and the
  patches before it);
- RMSNorm, then a SwiGLU block (a feed-forward block gated by a SiLU branch),
  `w_out(silu(w_in(h)) * v_in(h))`, of hidden width 688 (`--d-ff 688`).

A final RMSNorm closes the backbone.

**Head** (`Student.head`, `monotone`). `Linear(256, 1024)`, SiLU, `Linear(1024, 15 x 32)`: fifteen
knots for each of the 32 steps of the next patch. `monotone` keeps the median knot as produced,
builds each knot above it as its lower neighbour plus a softplus increment (`log(1 + e^x)`,
always positive) and each knot below as its upper neighbour minus one, so the levels never cross.
The levels (`student.LEVELS`, `model.json` `levels`): 0.01, 0.05, 0.1 to 0.9 in steps of 0.1,
0.95, 0.99, 0.995 and 0.999.

**Loss on the truth** (`train.loss_parts`). Pinball (quantile loss) `max(tau * e, (tau - 1) * e)`
with `e` the truth minus the knot, at all fifteen levels `tau`, over positions 0 to 62, masked to
the present targets and averaged over present points and levels. Its weight is 1; the
distillation term reaches the nine levels 0.1 to 0.9 only, so the six outer levels learn from the
truth alone. To the series' units (`forecast.next_patch`): `loc + scale * sinh(q)` with the
scaler at the end of the context, then the [floor](floor.md).

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the asinh space, lower is better; **dense** channels have a
value most of the time, **sparse** ones are mostly zero; **held-out time** is later tiles of
trained series, **unseen containers** every tile of series held out of training; a **mixture of
experts** (MoE) replaces the feed-forward block with several routed copies. **BOOMLET** is the
32-dataset subset of Datadog's BOOM observability benchmark, **MASE** the mean absolute scaled
error of the median relative to seasonal naive. The A40 and the RTX 2080 Ti are the NVIDIA cards
the runs used.

- Dense student from scratch: the first student of this shape, under that day's recipe on the
  first 30-second store, came within 3.8 % of the teacher's dense held-out-time Winkler in 20
  minutes; three seeds spread 0.2 to 0.3 % on dense.
- Capacity, 19 M against 4 M: a 19 M dense student ties the 4.28 M one on dense held-out time
  (+0.01 % on the A40, −0.39 % on the RTX 2080 Ti, inside 0.6 % seed noise) and is about 1.2 %
  better on unseen containers.
- Width, routing and data scaling: feed-forward width 344 costs nothing on held-out time
  (+0.19 % and −0.01 % on the two cards), 172 depends on the card (+0.17 %, +1.00 %), and every
  MoE scores like the dense block of its active width.
- Dense against MoE at 200,000 steps: dense leads the tagged MoE without a balance loss by 0.2
  to 0.4 % and the untagged MoE by 0.9 to 1.6 % on dense channels at every checkpoint from
  100,000 to 200,000 steps, on both cards.
- MoE with the resolution tag: both MoE arms trail the float32 dense control in all five
  held-out cells, +0.26 to +0.83 % dense and +1.3 to +2.6 % sparse, resolved per tile, at 2.7
  times the step; the MoE line is closed.
- BOOMLET students: the dense student's MASE was 0.740, every MoE arm 2 to 5 % behind dense.
- Depth, eight layers against four: two seeds; on the tail averages (the mean of each run's last
  three snapshots) BOOMLET CRPS is 0.91 and 0.83 % lower (chance four layers are better 10 and
  4 %), and sparse unseen-container pinball 7.47 and 1.12 % worse, resolved, at 1.74 times the
  parameters; eight layers are not in the published model.

## Open

- A flow-matching head (a generative head that samples the next patch) trained beside the
  quantile head moved the quantile head 2.2 % better on BOOMLET MASE and 0.5 % worse on our dense
  tiles at 100,000 steps, one seed.
