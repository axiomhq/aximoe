# Rollout: forecasts past the first patch

Since 2026-11-15 the release `aximoe-261115` is trained for this rollout over up to 30 patches ([deep unroll](deep-unroll.md)); the page holds for it.

Since 2026-11-23 the published model `aximoe-261123` rolls out in patches of 16 steps ([patch width](patch-width.md)) and is trained for it over up to 61 patches ([deep unroll](deep-unroll.md)); the page holds for it.

Median feedback is the rollout of every forecaster since the first runs on BOOMLET (Datadog's
public observability benchmark) of 2026-10-30, today `forecast.py` (the path BOOMLET runs
through) and the MCP (Model Context Protocol) server; the floor acts before the feedback since
2026-11-03. The release `aximoe-261105` runs it at the default `--paths 0`, since
2026-11-05 with each fed-back patch's band beside its median, in `forecast.py` and in the MCP
server ([band input](band-input.md)).

## What it does

The student forecasts one patch, 32 steps, per forward pass; every later patch is forecast from
a context that ends in fed-back values.

**`forecast.forecast`** (`forecast.py`, `--horizon N`):
1. The window is the last 2,016 steps of the input (`read_windows`; a shorter input is padded at
   the front with missing steps).
2. Each patch (`next_patch`): `causal_stats` over the current window, the student's knots at
   position 62, mapped to the series' units as `loc + scale * sinh(q)` with the scaler at the end
   of the window.
3. Floor: a window whose observed 2,016 steps hold no finite value below zero (`floor_eligible`,
   decided once) has every quantile of each patch clipped at zero (`floor_at_zero`) as the patch
   is produced ([floor](floor.md)).
4. Feedback: the floored patch's median (level 0.5) joins the window at its end, the oldest patch
   leaves, and the next patch is forecast from the new window.

Steps 0 to 31 are the direct forecast and later steps recursive; the layout file beside the
output (`<out>.json`) records the recursion used. A window with fewer than 32 observed steps, or
with none in its last 32, gets no forecast (`forecast.answerable`): its levels are NaN and the
layout counts it as `unanswered`. The public harnesses score such a window with its last observed
value and write the count beside the scores.

**`investigate.Model.forecast`** (the MCP server). The context is the 480 one-minute steps before
the scanned window (`Settings.ref_model`, 15 patches), the window 5 to 240 minutes (default 60),
so one to eight patches. The context sits at the start of a tile padded with missing steps; each
patch is read at the last context position with that position's scaler, floored, reported as its
0.1, 0.5 and 0.9 levels, and its floored median is appended: the context grows by one patch per
step, up to the student's 64 positions. A context with under half its minutes present, or with no
data in its last patch, gets no forecast (`context_gap`): the series is listed under `skipped` with
the reason, and a history window in that state stays out of the calibration.

**Alternatives in `forecast.py`.**
- `--paths N --seed S`: from the second patch on, N paths each feed back a draw from their own
  quantile function (`draw_from_knots`, linear between the levels, the end knots beyond them),
  one probability per path and patch shared by the patch's 32 steps; each step's levels are those
  of the equal-weight mixture of the N quantile functions (`mixture_quantiles`, the mean of their
  distribution functions, inverted). The first patch is the direct one. It costs N passes per
  recursive patch.
- `--hybrid`, with `--paths`: the median rollout runs beside the paths as one more path fed its
  0.5 level, and each step's mixture levels are shifted so that their 0.5 level is the median
  rollout's median: the paths' band around the median rollout's centre.
- `--widen <factors.json>`: after the rollout, each level's distance from the median is multiplied
  by a factor per forecast patch and level (`widen_band`), in the asinh space of the window's
  statistics at the forecast origin.
- A model trained with `--horizon-patches K` forecasts K patches from one pass, and their medians
  join the window together.
- A model trained with `--unroll D` ([band input](band-input.md)) carries each forecast patch's 0.1
  and 0.9 levels beside its median, as offsets the model reads; one pass per patch.

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the teacher's standardised space, lower is better;
**dense** channels have a value most of the time, **sparse** ones are mostly zero; **direct** is
the one-patch forecast from true context. **BOOMLET** is the 32-dataset subset of Datadog's BOOM
observability benchmark; **MASE** (mean absolute scaled error of the median) and **CRPS**
(continuous ranked probability score) are relative to seasonal naive over 90 scaled
configurations, paired per configuration with a 95 % interval.

- Drift curve: on the old 1-minute store the median rollout's dense Winkler is 1.41 times the
  direct one at patch 2 and 4.4 times at patch 32, worse than holding the last value from about
  step 600; sparse breaks at patch 3 (step 65); the bands keep their one-patch width at every
  depth.
- Resolution tag: the resolution row cut the sparse rollout Winkler at patch 3 by 62 % in one
  checkpoint pair; two seeds of one recipe differ by a factor of two there, so that was one
  checkpoint's draw.
- Generative head, BOOMLET: sixteen paths lower CRPS by 3.61 % (0.9395 to 0.9895) and raise MASE
  by 4.57 % (1.0252 to 1.0676); the hybrid keeps MASE (ratio 1.0000), lowers CRPS by 4.70 %
  (0.9356 to 0.9720) and moves the rank by CRPS from 12.42 to 10.81 of 24, with CRPS 21 % worse
  on the six low-variance configurations.
- Generative head, our tiles past the first patch: the median rollout's 0.1 to 0.9 band holds 26
  to 51 % of the dense truth where it should hold 80 %, the hybrid 79 to 82 %, with dense
  Winkler 41 % lower, alike at three draw seeds.
- Rollout bands: band factors fitted on our fleet lower dense Winkler past the first patch by 19
  to 21 % and multiply BOOMLET CRPS by 2.6 (3.8 to 4.6 from fifteen patches on); a four-patch
  head calibrates the band on our tiles at two seeds with a small, seed-dependent BOOMLET gain
  (CRPS -5.0 and -1.8 %); the band input at the full recipe with the tail average (the mean of
  the run's last three snapshots) lowers BOOMLET CRPS by 10.7 and 10.4 % at two seeds (rank by
  CRPS 12.4 to about 9.1) with the band past the first patch at 0.78 to 0.82, at one pass per
  patch.

## Open

- The hybrid is a no-go (2026-11-04): the MCP server has to reply in seconds, and sixteen paths
  make the forecast step 10 to 14 times slower. The median rollout's band past the first 32
  minutes holds a third to a half of normal behaviour. The band input fixes it at one pass per
  patch; the release `aximoe-261105` carries it, in the forecaster and in the MCP server.
- A rollout comparison needs three seeds a side; the reinforcement-learning stage with the drift
  curve as its baseline.
