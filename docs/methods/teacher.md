# Teacher: Toto 2.0 313m's quantiles and the distillation term

Since 2026-11-15 the release `aximoe-261115` also learns the teacher's forecasts of hidden patches: [far labels](far-labels.md). The rest of this page holds for it.

Since 2026-11-23 the published model `aximoe-261123` learns the teacher's forecasts on a 16-step grid, from a second pass of the teacher over every tile read from step 16 ([patch width](patch-width.md)). The rest of this page holds for it.

Toto 2.0 313m is the teacher since 2026-10-28 and the distillation weight is fixed at 1.0 since
2026-10-29; the release `aximoe-261105` (`model.json`: `"teacher": "Datadog/Toto-2.0-313m"`)
trained under both.

## What it does

**The teacher pass.** Toto 2.0 313m,
Datadog's pretrained forecaster with 312.69 M parameters, runs once over every tile of a store,
each tile alone as one variate with its missing steps masked, so it sees nothing the one-variate
student cannot. It ran in float32 with TF32 (TensorFloat-32, the reduced-mantissa tensor-core
format) off: the store's model name carries no `tf32` suffix. A self-check re-runs batches (four
by default) and fails unless they reproduce bit for bit. Per tile the store holds flat
little-endian arrays:

- `values.f32`: the raw series, 2,048 float32 values, NaN where missing;
- `quantiles.f16`: the nine quantiles at levels 0.1 to 0.9 for each of the 63 positions and each
  of the 32 steps of the patch a position forecasts, `[9, 63, 32]` in float16, in the scaler's
  asinh space (the inverse hyperbolic sine of the standardised value);
- `loc.f32`, `scale.f32`: the causal scaler's location and scale at the last step of each of the
  63 conditioning patches, float32, the scale at least 1e-6.

Position s forecasts patch s + 1 from patches 0 to s. The student reads the same location and
scale ([student](student.md)), so the two models' quantiles share one space.

**The distillation term** (`train.loss_parts`). For the student's knots at positions 0 to 62:

- the difference between the student's nine shared levels 0.1 to 0.9 (`student.SHARED`) and the
  teacher's nine, in the asinh space;
- the Huber distance with delta 1: half the squared difference below 1, the absolute
  difference less one half above it;
- masked to the target points that are present, summed, and divided by the present points times
  nine: the mean over points and levels.

The step's loss (`train.main`) is `truth + lambda * distill`; the balance and flow terms there are
zero for a dense student with the quantile head alone. `--lambda 1.0:1.0` runs the weight
linearly from the first value to the second over the run, so it is 1.0 at every step (train.log:
`lambda 1.0->1.0`). The truth term has weight 1 and covers all fifteen levels; the six outer
levels (0.01, 0.05, 0.95, 0.99, 0.995, 0.999) learn from the truth alone. The float16 quantiles
are promoted to float32 in the subtraction.

**What the teacher buys.** The model is used on series it has never seen, not on the fleet it
was trained on: series held out of training and BOOMLET, Datadog's public observability
benchmark, are the measures, and against training on the truth alone the teacher buys 3 to 5 %
on BOOMLET at equal steps.

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the asinh space, lower is better. **Dense** channels have a
value most of the time, **sparse** ones are mostly zero. **Held-out time** is later tiles of
trained series, **unseen containers** every tile of series held out of training. **BOOMLET** is
the 32-dataset subset of Datadog's BOOM observability benchmark; **MASE** (mean absolute scaled
error of the median) and **CRPS** (continuous ranked probability score) are relative to seasonal
naive over its 90 scaled configurations.

- Teacher choice: the 313m scored best of the Toto 2.0 sizes on one-variate tiles of the first
  30-second store.
- Compute and precision: TF32 moves Toto's zero-shot Winkler 20 to 50 times less than the label
  effect (the change from telling Toto the channel type) and bfloat16 as much as that effect;
  the pass runs once, in float32.
- Distillation weight: on the old 1-minute store at 200,000 steps, fixed 1.0 held dense
  held-out-time Winkler at 1.5125 against the teacher's 1.4805 and fixed 0.2's 1.5473; 0.2 was
  better on held-out containers (1.5976 against 1.7023); the anneal from 1.0 to 0.2 walked
  between them and was dropped.
- Truth only at 100,000 steps: weight 0 against its twin of equal seed and steps is +0.46 %
  (+0.0045 to +0.0119) on dense held-out time, −0.36 % (−0.0129 to −0.0019) dense and −8.6 %
  sparse on unseen containers, and +4.82 % BOOMLET MASE (1.0225 to 1.0768), +3.18 % CRPS (1.0124
  to 1.0521).
- What the teacher buys: the reading across records: breadth on foreign data (2 to 5 % on
  BOOMLET at equal steps) and 0.4 % forward in time, at a cost on unseen containers; 2.27 times
  the steps buy 0.1 to 0.4 % instead.
- Chronos-2 for the outer levels: as a second teacher on the levels 0.01, 0.05, 0.95 and 0.99 at
  100,000 steps it made BOOMLET CRPS 4.97 % worse and the outer levels 3.6 to 32.6 % worse;
  measured, not adopted (2026-11-06).

## Open

- Weights 0.2 and 0.5 at 100,000 steps on the store of record, to find where the unseen-container
  gain and the forward-in-time loss cross.
- A weight above 1 or pure distillation.
- The truth-only recipe at full length with the tail average (the mean of the run's last three
  snapshots).
