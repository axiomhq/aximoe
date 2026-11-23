# Training: optimiser, schedule, precision and seed

Since 2026-11-15 the release `aximoe-261115` trains under a cosine decay for 1,043,796 steps ([schedule](schedule.md)) with a deeper unroll ([deep unroll](deep-unroll.md)). What follows describes `aximoe-261105`.

Since 2026-11-23 the published model `aximoe-261123` trains for 1,175,000 steps ([schedule](schedule.md)) in half precision: `--dtype fp16` runs the passes under float16 autocast with loss scaling, and the unroll's passes and the training pass replay as CUDA graphs. Seed 1.

Batch 64 at a learning rate of 1.2e-3 after 200 warm-up steps since 2026-10-28, the compiled
PyTorch trainer since 2026-10-31. The release `aximoe-261105` is, since
2026-11-05, a run of seed 2, trained 2026-11-05 on the A40 (an
NVIDIA A40 card) with the band input ([band input](band-input.md)) and then averaged over three
snapshots (the tail average: every tensor is the mean of its values at steps 250,000, 300,000
and the final 347,932). Its command, run from the repository's root:

    python python/aximoe/train.py \
        --store data/derived/<store>/teacher-313m \
        --batch 64 --lr 1.2e-3 --warmup 200 --steps 347932 --eval-every 10000 --eval-tiles 2048 \
        --eval-seed 0 --save-every 50000 --seed 2 --lambda 1.0:1.0 --width-shares 10:1,30:1,60:1 \
        --unroll 3 --save data/models/<run>/seed2

Every other setting is a default of `train.parse`: `--layers 4`, `--d-ff 688`, `--experts 0`,
`--tags on`, `--resolution-tags on`, `--tag-dropout 0.1`, `--compile on`, `--dtype f32`,
`--eval-batch 256`. The first lines of `train.log` print the store, the model's settings, the
optimiser's and the draws per width.

## What it does

All in `train.main`:

- **Optimiser:** AdamW (Adam with decoupled weight decay) over every parameter, betas 0.9 and
  0.999, eps 1e-5, weight decay 0.01.
- **Learning rate:** `lr * min((step + 1) / 200, 1)`, a linear warm-up over the first 200 steps
  to 1.2e-3, then constant to the last step.
- **Batch:** 64 tiles per step from the sampler ([data](data.md)).
- **Gradient clipping:** the global gradient norm clipped at 1.0 (`clip_grad_norm_`).
- **Length:** 347,932 steps; 154.5 minutes at 37.6 steps/s on the A40 with the band input's unroll
  (train.log), against 77.3 minutes at 75.0 steps/s without it. The unroll on the cached pass
  runs the same recipe at 48.5 steps/s over the first 20,000 steps against 37.5, 119.6 minutes
  projected for the full run ([band input](band-input.md#cost)).
- **Precision:** float32; TF32 (TensorFloat-32, the reduced-mantissa tensor-core format) off for
  matrix products and for cuDNN, NVIDIA's kernel library (`allow_tf32 = False`); `--dtype f32`
  runs without autocast.
- **Compilation:** `torch.compile` wraps `model.forward_full` and `loss_parts`, and with
  `--unroll` also `model.cached_pass` and `model.next_position`; gradient clipping and the
  optimiser step run outside it.
- **Seed** (`--seed 0`): `torch.manual_seed` (the initial weights), the sampler's generator, and
  the tag-dropout generator at the seed plus one. The evaluation sample has its own `--eval-seed
  0`, so seed 1 was scored on the same tiles.
- **Snapshots:** `student-<step>.pt` every 50,000 steps and `student.pt` at the end, beside
  `model.json` (`student.describe`); the average of `aximoe-261105` reads steps 250,000, 300,000
  and the final 347,932.
- **Evaluation during training** (`train.evaluate`): every 10,000 steps the 2,048 held-out-time
  tiles of evaluation seed 0 are scored for the student, the teacher (from the store) and
  persistence (the last value carried forward): Winkler and pinball (defined below) and the share
  of targets beyond each outer level; at the end the three splits. The final weights on held-out
  time: dense Winkler 1.7700 against the teacher's 1.7606 and persistence's 5.6469, sparse 0.1488
  against 0.1689 and 0.3906.

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the teacher's standardised space and **pinball** the
quantile loss, lower is better; **dense** channels have a value most of the time, **sparse** ones
are mostly zero. **BOOMLET** is the 32-dataset subset of Datadog's BOOM observability benchmark,
**MASE** the mean absolute scaled error of the median relative to seasonal naive over its 90
scaled configurations. **Paired** means per tile or per configuration with a 95 % interval.

- Batch size on 1-minute data: at equal A40 wall-clock, batch 64 at 1.2e-3 (the rate scaled
  linearly with the batch) beat batch 16 by 1.4 to 2.9 % on dense cells and 1.7 to 4.5 % on
  sparse, two seeds each.
- bf16 autocast: at 100,000 steps bfloat16 was 17 % faster and worse in every cell, dense
  held-out time +0.15 % (+0.0010 to +0.0037 Winkler), sparse +0.62 %; float32 stayed for that
  recipe.
- Exposure arm: 2.27 times the steps on an unchanged two-width recipe buy 0.09 to 0.40 % on
  dense tiles, cost 0.6 to 4.9 % on sparse tiles and leave BOOMLET MASE at +0.15 % (0.9938 to
  1.0097).
- Seed spread: paired, two checkpoints of one recipe differ by 0.1 to 0.3 % dense and 0.8 to
  2.9 % sparse with intervals clear of zero, and three seeds spread 3.3 % on BOOMLET; a recipe
  claim below that needs several same-card pairs.
- Cross-card spread: the same seed on the A40 and the RTX 2080 Ti lands 0.05 to 1.0 % apart on
  dense, 0.2 to 3.6 % on sparse and about 2 % on BOOMLET MASE.
- Three widths, equal shares, second seed: seed 1 minus seed 0 is −0.04 to −0.43 % on dense
  Winkler and +0.2 to −2.0 % on sparse, and +1.38 % on BOOMLET MASE (+0.07 to +2.81 %); seed 0
  was the one released (`aximoe-261103` is its tail average).

## Open

- Batch 128 at equal wall-clock.
