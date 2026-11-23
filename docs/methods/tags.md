# Tags: the channel row and the resolution row

Since 2026-11-15 the release `aximoe-261115` has resolution rows for ten widths from 10 seconds to 1 hour ([coarse widths](coarse-widths.md)); the rest of this page holds for it.

Since 2026-11-23 the published model `aximoe-261123` has the same rows; its published BOOMLET number picks the channel row per window, with the `unknown` row reported beside it.

The channel row with its `unknown` row and 0.1 dropout since 2026-10-28, one row per channel
decided 2026-11-01, the resolution row since 2026-10-31; both in the release `aximoe-261105`
(`model.json` config: `backbone_tags` and `resolution_tags` true).

## What it does

**Two tables** (`student.Student`, `batches.Tags`) of vectors of width 256; one row of each is
added to every patch embedding of a tile (`Student.hidden`).
- **Channel table**, 38 rows (`model.json` config `tags`): one per channel code of the store's
  index, 37 in index order, plus `unknown`, row 37. `model.json` `channels` maps the 36 channels
  with training tiles to their rows. Row 8 belongs to the held-out channel `fs_write_bytes_rate`;
  it receives no gradient, and that name, like any outside `channels`, maps to `unknown`.
- **Resolution table**, 4 rows (config `resolutions`): one per training bin width, `widths`
  10, 30 and 60 seconds, plus `unknown`, row 3.

A tile's two rows travel as one packed id, the channel row plus (resolution row + 1) times 65,536
(`student.TAG_BASE`, `student.tag_id`, `batches.pack_tags`).

**Dropout in training** (`batches.make_batch`, `--tag-dropout 0.1`). For each training tile the
channel row and, independently, the resolution row become `unknown` with probability 0.1, drawn
from the generator at the run's seed plus one, so both `unknown` rows train on every channel and
width.

**At inference.**
- `forecast.py`: `--tag unknown`, the default, gives every window the `unknown` channel row;
  `--tags FILE` reads one channel name per window, a name with a row in `channels` gets that row,
  any other name `unknown`; `--tag auto` is the in-context pick below. `--resolution SECONDS`
  picks the width's row; a width outside 10, 30 and 60, or none, gets `unknown`. The harness of
  BOOMLET, Datadog's public observability benchmark, passes each
  dataset's bin width as `--resolution`; it runs with `--tag unknown` (the resolution tag alone) and with
  `--tag auto`.
- The MCP (Model Context Protocol) server (`investigate.py`, `Model.forecast`): each of its five
  metrics carries its channel name from the tile index (`cpu_usage_rate`, `cpu_throttled_rate`,
  `mem_working_set`, `restarts_rate`, `oom_rate`, all five with trained rows), at the 60-second
  resolution row (`Model.BIN_SECONDS`).
- In-context pick (`forecast.pick_tags`, `tagpick.candidates`, `tagpick.picks`): the student
  forecasts the window's own patches in context under each of 37 candidates, the 36 channel rows
  and `unknown`, at the window's resolution row, and keeps the row with the lowest pinball
  (quantile loss) per present point over the window.

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the teacher's standardised space, lower is better;
**dense** channels have a value most of the time, **sparse** ones are mostly zero; **in
distribution** means our own held-out tiles: **held-out time** (later tiles of trained series),
**unseen containers** (series held out of training) and the **unseen channel**
(`fs_write_bytes_rate`). BOOMLET has no channel names; **MASE** is the mean absolute scaled error
of the median relative to seasonal naive over its 90 scaled configurations, **MAE** the mean
absolute error on its **6 unscaled**, low-variance ones. A mixture of experts (**MoE**) routes
each token to one of several feed-forward blocks.

- Tags on against off: the channel row gains 4 to 5 % Winkler on sparse channels at a cost of
  0.2 to 0.7 % on dense; an untrained row on the unseen channel costs 4.7 % and 13 % in two
  seeds.
- Unknown tag: the `unknown` row trained by dropout 0.1 puts the unseen channel 1.1 % behind
  tags off instead of 4.7 to 13 %, keeping the sparse gain.
- Tag as a token: a leading tag token, alone or beside the row, is worse in five cells of five,
  inside the single-run spread; the row stays.
- Channel tags: the decision across records; training without tags is 2.86 % worse on sparse
  held-out time and 0.05 % better on dense, so the tag's value is in training, while at
  inference the name adds 0.05 to 0.5 %.
- Tag groups: five name families as rows are 9.8 % worse on sparse held-out time and 4.2 % on
  BOOMLET MASE, behaviour groups 1.7 to 3.8 % worse, all resolved; nothing from 36 rows down to
  one beats 36.
- Channel taggers: a trained tagger names 18 to 20 % of channels; under its names the student
  gains at most 0.05 % in distribution and ties `unknown` on BOOMLET, 1.1 % behind the
  in-context pick.
- Tag value and the in-context pick: in distribution `unknown` is 0.03 to 0.43 % worse than the
  true row, resolved; on BOOMLET the pick gains 0.31 % MASE (0.10 to 0.54 %) on the 90 and 4.0 %
  MAE (2.3 to 5.6 %) on the 6 unscaled, at 2.3 times the forecast time.
- Resolution tag: paired, the row gains 0.24 and 0.27 % on unseen-container dense cells in both
  seed pairs and flips sign elsewhere; on BOOMLET it is a null at inference (ratio 1.0007,
  0.9987 to 1.0025).
- MoE tags in router and backbone: the channel tag in an MoE router changes the routing but not
  the quality, and at equal steps there is no gap to dense for it to close.

## Open

- The dropout rate, 0.1 since the first day.
- The tag's value against context length, the MCP server's 480 steps included.
- Rows finer than the channel, and tags carrying unit, instrument type and bounds.
- The oracle row on BOOMLET, the bound on what a better pick adds; `unknown` stays the
  forecaster's default until it is known.
