# Band input: the forecast's band fed back beside its median

Since 2026-11-15 the release `aximoe-261115` unrolls deeper and with frozen statistics: [deep unroll](deep-unroll.md), [frozen feedback](frozen-feedback.md). What follows describes `aximoe-261105`.

Since 2026-11-23 the published model `aximoe-261123` reads the same input beside patches of 16 steps ([patch width](patch-width.md)).

Measured 2026-11-04 and 2026-11-05; in the release `aximoe-261105` since 2026-11-05, seed 2,
with the band carried by `forecast.py` and by the MCP server's rollout. The full recipe with the
band input runs `train.py` with the recipe of record plus `--unroll 3` (on the A40: 347,932
steps, then the tail average, the element-wise mean of the run's last three snapshots).

## What it does

Past the first patch the median rollout ([rollout](rollout.md)) feeds back each forecast patch's
median as if it were observed. The model cannot tell a fed-back patch from data, and a fed-back
median is smoother than any real patch, so the model reads a calm recent history and forecasts a
narrow band: on our tiles the band past the first patch holds 26 to 51 % of the truth where it
should hold 80 %. The band input tells the model which context steps are its own forecasts and
how uncertain they were.

**The input.** `StudentConfig.band_input` widens each input patch from two rows of 32 (the
standardised values and the present-value mask) to four: the 0.1 and the 0.9 levels as offsets
from the value, in the same asinh space as the value. An observed step carries offsets of zero,
a fed-back step the band it was forecast with. The patch embedding's first layer
(`Student.embed1`) grows from 64 to 128 inputs, 65,536 weights; the model has 4,351,840
parameters against 4,286,304. Nothing else in the model changes. `Student.hidden`,
`forward_full` and `forward` take the band as an optional argument; without it the offsets are
zero, so every caller that forecasts from data alone (the trainer's evaluation, the paired comparison
on tiles, the in-context pick) runs unchanged.

**Training: the unroll** (`train.py --unroll D`, which also sets `band_input`; `unroll` in
`train.py`). Each batch is first rolled out on itself, without gradients:
1. Per tile a depth `j` is drawn uniformly from 0 to `D` (a torch generator seeded with the
   training seed plus 2), and the last `j` context patches are to be replaced: the cut is
   position `62 - j`, so the forecast patches sit at the end of the context, as at inference.
2. For `t` from 1 to `D`, the tiles with `j >= t` forecast the next patch from position
   `cut + t - 1`. Pass 1 runs the whole batch and keeps each layer's attention input
   (`Student.cached_pass`); each later pass runs only the position whose patch was just
   replaced, one per tile, against those inputs (`Student.next_position`), which under causal
   attention is a full pass over the batch as it stands up to float rounding (fed-back inputs
   within 8.64e-06 of full passes on the A40). The forecast's 0.1, 0.5 and 0.9 levels are
   mapped from the asinh space of that position's statistics to the raw scale and back into the
   asinh space of the replaced patch's statistics. The median becomes the patch's input, the mask
   is set to 1, and the two offsets become its band.
3. The training pass then runs on the modified batch. The truth term (pinball over all fifteen
   levels against the true next patch) is unchanged at every position. The teacher term is
   masked (`distill_mask`) at every position whose context holds a forecast: Toto saw the true
   values there, so those positions learn from the truth alone.

`D = 3` covers the MCP server's 60 and 120-minute windows, whose last forecast patch has up to
three fed-back patches before it. BOOMLET's medium and long terms run 15 to 29 patches, past
anything the unroll shows; the gains there are the extrapolation of a depth-3 unroll.

**Inference** (`forecast.forecast`, `next_patch`, `band_offsets`). The rollout keeps, beside its
window of values, the raw 0.1 and 0.9 levels of every step: missing for observed steps, the
floored levels of each forecast patch once it is fed back. `next_patch` turns them into offsets
in each input patch's asinh space, with the window's own causal statistics, and passes them to
the model. One pass per patch, as for the median rollout. Not combinable with `--paths` or with
a head of several patches.

**The statistics approximation.** In training, a replaced patch is standardised with the store's
statistics, computed on the true values; at inference the window's statistics are computed over
the window as it stands, fed-back medians included. Both are running means and deviations from
the start of the window, so one patch moves them little; the size of the difference is not
measured.

## Cost

- Training: on the A40 the recipe runs at 48.5 steps/s over its first 20,000 steps, 119.6
  minutes projected for 347,932 steps, against 75.0 steps/s and 77.3 minutes without the band
  input (measured 2026-11-07). The unroll with a full pass per replaced patch, which trained
  the release `aximoe-261105`, ran the same 20,000 steps at 37.5 steps/s and the full
  run in 154.5 minutes.
- Inference: one pass per patch plus the offsets. On the 2080 Ti, 2,048 tiles over 4 patches take
  2.3 s against 2.0 to 2.2 s for the release `aximoe-261103`. In the MCP server on the CPU
  (`investigate.Model.forecast`, five investigations over MCP against the median rollout of
  `aximoe-261103`), the aximoe stage took 0.79 s against 0.74 s for 8 series, 103.0 s
  against 103.8 s for 842 series over 60 minutes, 117.8 s against 115.3 s for 842 series over
  240 minutes, 69.8 s against 68.7 s for 536 series, 0.45 s against 0.40 s for 4 series.

## Why

Terms as in the [rollout](rollout.md) page; **inside** is the share of the truth within the 0.1 to
0.9 band (nominal 0.80); the BOOMLET resolution tag is the dataset's bin width on the resolution
row with the channel row `unknown`.

- At the full recipe with the tail average, against the release `aximoe-261103` (the thirds
  seed 0 tail average), three seeds: BOOMLET CRPS 10.72, 10.05 and 10.31 % lower, MASE 2.18,
  1.46 and 1.97 % lower, rank by CRPS 12.42 to 9.04, 9.18 and 8.98 of 24; past the first
  patch dense Winkler 47.7 to 49.8 % lower with the band inside 0.78 to 0.82; sparse unseen
  containers on the next patch pinball +7.90, +9.14 and +7.55 %. Seed 2 is first or second of
  the three on each of fifteen cells (BOOMLET's five columns, six next-patch cells, four past
  the first patch), first on ten.
- At the full recipe with the tail average, against the thirds tail average of the same seed
  (seed 0: the release `aximoe-261103`), two seeds: BOOMLET CRPS 10.72 and 10.40 % lower
  (intervals 0.8720 to 0.9106 and 0.8754 to 0.9146), MASE 2.18 and 1.31 % lower (the second
  interval touching 1), rank by CRPS 12.42 to 9.04 and 12.44 to 9.18 of 24. On our tiles past
  the first patch dense Winkler 46 to 50 % lower and the band inside 0.78 to 0.82 where the
  reference holds 0.33 to 0.42; the dense next patch unchanged (+0.09 % at most). Costs:
  BOOMLET 10 s MASE +1.50 % at seed 1 (resolved; -0.72 % at seed 0), and sparse unseen
  containers on the next patch, pinball +7.90 % at seed 0 and +1.39 % at seed 1 against the
  averaged reference.
- Without averaging, raw final weights against the thirds raw final weights: BOOMLET CRPS 10.52
  and 11.38 % lower, and the sparse unseen-container cost gone (seed 0 pinball -0.41 %, seed 1
  +0.46 % unresolved). The thirds model's own average gained 8.8 % pinball on that cell at
  seed 0 and 0.4 % at seed 1; the band model's average about 1 % at either.
- The tail average helps the band model at the full recipe: BOOMLET CRPS 0.74 and 1.04 % lower
  than its raw weights. At 100,000 steps the average of the 50,000 and 100,000 snapshots hurt it
  (CRPS +1.38 %, MASE +2.05 %).
- At 100,000 steps, seed 0, against the thirds run's own snapshot: BOOMLET CRPS 12.49 % lower,
  MASE 4.18 % lower, the next patch unchanged.

## Tests

- `test_train.Unroll`: the unroll leaves the truth context up to the cut untouched and without
  band, and the first forecast patch is the prediction at the cut, mapped into the replaced
  patch's statistics, as input and band.
- `test_train.DistillMask`: a teacher mask of ones reproduces the loss exactly; a partial mask
  drops positions from the teacher term only.
- `test_forecast.BandInput`: the rollout feeds each forecast patch's band back, and the second
  patch differs without it.
- `test_investigate.BandRollout`: the MCP server's rollout gives the second patch that
  `forecast.next_patch` gives with the first patch's median and band fed back, and a different
  one without the band.

## Open

- The MCP server's hits move with the band: in five investigations the aximoe hits from the
  first half hour on fell (465 to 378 over a 240-minute window) and those in the first half hour
  rose, since a bin's rarity is ranked against the previous day's bins of the whole window, whose
  late bins now score lower. Which hits are right is not measured.
- What part of the gain is the band channel and what part training on fed-back patches: an unroll
  without the band input (medians alone) separates them.
- The sparse unseen-container cost against the averaged reference, and BOOMLET's 10-second slice.
