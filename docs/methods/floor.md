# Floor at the raw zero

Kept 2026-11-03. The forecaster (`python/aximoe/forecast.py`, the path BOOMLET runs
through) and the MCP server's rollout (`python/aximoe/investigate.py`) apply it;
`forecast.forecast(..., floor=False)` returns the model's own quantiles. The measurements below
ran through scripts outside the package; on `aximoe-261103` the
package's floor reproduces the BOOMLET wrapper bit for bit.

## What it does

Most metrics the student forecasts cannot go below zero: counters, byte counts, OOM kills,
restarts, throttled periods. The student does not know that, and on long runs of zeros its band
can sit wholly below zero. The floor clips every forecast quantile at the raw zero for a series
whose observed context holds no value below zero.

- **Eligibility.** A window is eligible when no finite value of its observed context is below 0.
  Missing steps do not count.
- **Where it acts.** On every quantile of every forecast step of an eligible window:
  `q = max(q, 0)` in the series' units. In the standardized space the student and the reference
  scorer work in, the raw zero sits at `z0 = asinh((0 - loc) / scale)` with the tile's location
  and scale, and the floor is `max(q, z0)`.
- **Rollout.** Forecasts longer than one patch (32 steps) feed each patch's median back as
  context. The floor clips each patch as it is produced, so the median fed back is the floored
  one, and eligibility is decided once, from the observed context only.
- **Parameters.** None. There is nothing to fit, so no test set can be overfitted by it.

**Why it cannot hurt on a series that stays non-negative.** Take a truth `y >= 0`.

- Pinball loss of a quantile `q < 0` at level `tau` is `tau * (y - q)`. Raising `q` to 0 makes
  `y - q` smaller, so the loss falls.
- The Winkler score of a band `[l, u]` is its width `u - l` plus `2 / alpha` (here 10) times the
  distance by which `y` falls outside it. If `l < 0 <= u`, raising `l` to 0 narrows the band, and
  `y` cannot fall below the new edge. If `u < 0`, both edges rise to 0: the width drops to 0 and
  the miss distance drops from `y - u` to `y`.

So the floor is free wherever the truth stays non-negative. It costs only on a series whose
context is non-negative and whose future goes below zero.

## Where it came from

On 2026-11-03 the published model was the thirds model, seed 0:
three sampling intervals (10 s, 30 s, 1 min), each a third of the training draws, 347,932 steps.
On sparse held-out tiles it was about 1 % worse than its predecessor, and that had been put
down to a band "a hair wider" on runs of zeros. The diagnosis found the opposite.

The test cases were the positions on sparse held-out-time tiles of the store of record whose
input ends in a run of 16 to 62 all-zero patches and whose next patch is all zero. There are
40,572 of them, 62.9 % of all sparse positions. On these, the share of positions where the
0.1 to 0.9 band misses the zero truth:

| Model                                                    | inside the band | band wholly below zero | band wholly above zero |
| :------------------------------------------------------- | --------------: | ---------------------: | ---------------------: |
| two-width model, seed 0 (published 2026-11-01 to 11-02)  |          92.2 % |                  4.4 % |                  3.4 % |
| thirds model, seed 0 (published 2026-11-02 to 11-03)     |          86.6 % |                 12.1 % |                  1.4 % |
| thirds model, seed 1                                     |          96.8 % |                  3.3 % |                  0.0 % |
| thirds model, seed 0, 3-snapshot average (aximoe-261103) |          97.4 % |                  2.6 % |                  0.0 % |
| thirds model, seed 1, 3-snapshot average                 |          96.7 % |                  3.3 % |                  0.0 % |

Seed 0's band was not wider. It sat below zero on one zero-run position in eight. The tail
average (the element-wise mean of the run's last three snapshots, the 3-snapshot average of the
table) removes most of that by itself, and the floor removes the rest.

Before the floor, a fix inside the network was tried, in the style of weight-space programming
(rikiyat.github.io/blog/weight-space-programming).

- **The edit.** A rank-one change to the output layer along the direction that separates
  zero-run activations from all others, with its strength chosen on train tiles. It changed
  3.8 % of the weights.
- **Its result on held-out tiles.** Sparse Winkler fell by 0.04 to 0.33 %. Dense cells moved by
  -0.06 to +0.03 %, about as many worse as better.
- **Why it could not do more.** The learned direction leaks: on 10 % of dense positions it reads
  more than half its zero-run value. But even an exact zero-run gate in its place reached only
  -0.02 % on train tiles, so the limit is the edit's form, a fixed shift of the band on zero
  runs, more than its gate.
- **Against the floor.** On sparse cells the floor alone gains 17 to 47 times as much, and the
  edit with the floor on top trails the floor alone on the point estimate of every sparse cell.
  The edit was dropped.

## How it was tested

**Held-out tiles.** The floored and the plain quantiles are scored on the same held-out sample
that the paired comparison on tiles draws, paired per tile.

- **The sample.** Evaluation seed 0; 2,048 tiles per split, 1,024 for the unseen-channel split.
  It reproduces the paired comparison to the printed digit (checked on the store of record's held-out-time
  split).
- **The four stores.** The store of record, its 10-second store, its 1-minute store and the
  late 30-second store.
- **Splits.** Held-out time, unseen containers and, where the store has one, unseen channel.
- **Models.** Both seeds of the thirds model, on their final weights and on their 3-snapshot
  averages.
- **Statistics.** The 95 % interval comes from a paired bootstrap over tiles (4,000 resamples).
- **Eligibility.** 100.0 % of positions on every store: no context in these stores holds a
  negative value.

**BOOMLET.** BOOMLET is the 32-dataset subset of Datadog's BOOM benchmark, scored the way the BOOM
leaderboard scores it.

- **Fixed in advance.** The floor's definition above was fixed before the run, and nothing was
  tuned afterwards.
- **Harness.** A wrapper replaces `forecast.forecast` at run time and leaves the package
  untouched. It ran on the A40 with batch 64 and 3 workers.
- **Channel row, two ways.** Resolution tag: every window on the `unknown` channel row at its sampling
  interval's row, the published setting. Pick: each window takes the channel row with the best
  in-context fit, `--tag auto`.
- **Pairing.** Each floored run is paired per configuration against the same weights without the
  floor.
- **Eligibility.** 1,383 of 39,092 forecast windows (3.5 %), in 42 of 96 configurations. Every
  other window has a value below zero in its 2016-step context.

## Results

Terms:
- **Tile:** one 2,048-step window of one series.
- **Winkler:** the interval score of the 0.1 to 0.9 band, computed in the scorer's standardized
  space; lower is better.
- **Dense channels** have a value most of the time; **sparse channels** are mostly zero.
- **Splits:** held-out time is later tiles of trained series; unseen containers are tiles of
  series held out of training; an unseen channel is a channel never trained on.

**Store of record, seed 0 final weights.** A is the plain output, B the floored one; a negative
difference means the floor is better.

| Split and bucket         |      A |      B |    B minus A (percent; 95 % interval) |   B better on |
| :----------------------- | -----: | -----: | ------------------------------------: | ------------: |
| held-out time dense      | 1.7700 | 1.7664 | −0.0036 (−0.20 %; −0.0042 to −0.0030) | 29 % of tiles |
| held-out time sparse     | 0.1488 | 0.1424 | −0.0064 (−4.33 %; −0.0077 to −0.0054) | 98 % of tiles |
| unseen containers dense  | 2.0082 | 2.0024 | −0.0057 (−0.29 %; −0.0073 to −0.0044) | 28 % of tiles |
| unseen containers sparse | 0.1414 | 0.1314 | −0.0100 (−7.09 %; −0.0129 to −0.0076) | 99 % of tiles |
| unseen channel (sparse)  | 0.6558 | 0.6434 | −0.0123 (−1.88 %; −0.0162 to −0.0089) | 84 % of tiles |

On dense tiles the floor touches about a quarter of the tiles: those with a quiet stretch at
zero. The rest are unchanged.

**Every cell, both seeds.** Each cell is the change in Winkler against that seed's plain final
weights, in percent of A, with its 95 % paired interval in percent. "Average + floor" applies the
floor to the 3-snapshot average and is also measured against the plain final weights. One table
per seed; the reference column is that seed's plain final weights, Winkler.

Seed 0, against seed 0's plain final weights:

| Store                               | Split, bucket             | seed 0 final weights (reference) |           floor, seed 0 |  average + floor, seed 0 |
| :---------------------------------- | :------------------------ | -------------------------------: | ----------------------: | -----------------------: |
| store of record (10 s, 30 s, 1 min) | held-out time, dense      |                           1.7700 |  −0.20 (−0.24 to −0.17) |   −0.45 (−0.51 to −0.40) |
| store of record (10 s, 30 s, 1 min) | held-out time, sparse     |                           0.1488 |  −4.33 (−5.17 to −3.63) |   −5.53 (−7.12 to −4.30) |
| store of record (10 s, 30 s, 1 min) | unseen containers, dense  |                           2.0082 |  −0.29 (−0.36 to −0.22) |   −0.54 (−0.67 to −0.43) |
| store of record (10 s, 30 s, 1 min) | unseen containers, sparse |                           0.1414 |  −7.09 (−9.12 to −5.37) |  −9.03 (−11.88 to −6.65) |
| store of record (10 s, 30 s, 1 min) | unseen channel, sparse    |                           0.6558 |  −1.88 (−2.47 to −1.36) |   −2.67 (−3.69 to −1.83) |
| 10 s only                           | held-out time, dense      |                           1.9732 |  −0.22 (−0.27 to −0.18) |   −0.56 (−0.66 to −0.48) |
| 10 s only                           | held-out time, sparse     |                           0.1419 |  −3.98 (−4.23 to −3.74) |   −4.58 (−4.93 to −4.30) |
| 10 s only                           | unseen containers, dense  |                           2.1470 |  −0.39 (−0.52 to −0.28) |   −0.65 (−0.85 to −0.48) |
| 10 s only                           | unseen containers, sparse |                           0.1044 |  −7.78 (−9.48 to −6.42) | −10.16 (−12.84 to −7.85) |
| 10 s only                           | unseen channel, sparse    |                           0.6543 |  −1.31 (−1.67 to −1.04) |   −1.57 (−2.17 to −1.16) |
| 1 min only                          | held-out time, dense      |                           1.5427 |  −0.27 (−0.36 to −0.19) |   −0.59 (−0.71 to −0.49) |
| 1 min only                          | held-out time, sparse     |                           0.1639 |  −4.48 (−5.49 to −3.66) |   −5.47 (−6.77 to −4.33) |
| 1 min only                          | unseen containers, dense  |                           1.6521 |  −0.13 (−0.18 to −0.10) |   −0.50 (−0.62 to −0.40) |
| 1 min only                          | unseen containers, sparse |                           0.1562 |  −7.54 (−9.35 to −6.02) |  −9.47 (−11.78 to −7.30) |
| 30 s, Aug to Oct                    | held-out time, dense      |                           1.5478 |  −0.22 (−0.27 to −0.17) |   −0.48 (−0.57 to −0.40) |
| 30 s, Aug to Oct                    | held-out time, sparse     |                           0.1392 |  −5.75 (−6.97 to −4.74) |   −7.18 (−8.91 to −5.75) |
| 30 s, Aug to Oct                    | unseen containers, dense  |                           1.7142 |  −0.25 (−0.33 to −0.19) |   −0.57 (−0.70 to −0.46) |
| 30 s, Aug to Oct                    | unseen containers, sparse |                           0.1498 | −8.11 (−10.55 to −5.94) | −11.18 (−15.49 to −7.74) |

Seed 1, against seed 1's plain final weights:

| Store                               | Split, bucket             | seed 1 final weights (reference) |          floor, seed 1 | average + floor, seed 1 |
| :---------------------------------- | :------------------------ | -------------------------------: | ---------------------: | ----------------------: |
| store of record (10 s, 30 s, 1 min) | held-out time, dense      |                           1.7679 | −0.17 (−0.21 to −0.14) |  −0.37 (−0.45 to −0.30) |
| store of record (10 s, 30 s, 1 min) | held-out time, sparse     |                           0.1467 | −3.31 (−4.29 to −2.52) |  −3.34 (−4.50 to −2.45) |
| store of record (10 s, 30 s, 1 min) | unseen containers, dense  |                           2.0051 | −0.29 (−0.37 to −0.22) |  −0.44 (−0.53 to −0.36) |
| store of record (10 s, 30 s, 1 min) | unseen containers, sparse |                           0.1393 | −5.97 (−8.04 to −4.16) |  −6.07 (−8.33 to −4.16) |
| store of record (10 s, 30 s, 1 min) | unseen channel, sparse    |                           0.6543 | −1.79 (−2.43 to −1.24) |  −1.85 (−2.54 to −1.27) |
| 10 s only                           | held-out time, dense      |                           1.9701 | −0.23 (−0.29 to −0.17) |  −0.40 (−0.47 to −0.35) |
| 10 s only                           | held-out time, sparse     |                           0.1390 | −2.63 (−2.88 to −2.45) |  −2.49 (−2.73 to −2.30) |
| 10 s only                           | unseen containers, dense  |                           2.1462 | −0.42 (−0.55 to −0.29) |  −0.53 (−0.68 to −0.40) |
| 10 s only                           | unseen containers, sparse |                           0.1024 | −6.27 (−8.11 to −4.79) |  −6.41 (−8.30 to −4.79) |
| 10 s only                           | unseen channel, sparse    |                           0.6519 | −1.11 (−1.47 to −0.83) |  −1.17 (−1.53 to −0.89) |
| 1 min only                          | held-out time, dense      |                           1.5393 | −0.22 (−0.32 to −0.15) |  −0.38 (−0.49 to −0.27) |
| 1 min only                          | held-out time, sparse     |                           0.1614 | −3.44 (−4.46 to −2.60) |  −3.29 (−4.28 to −2.48) |
| 1 min only                          | unseen containers, dense  |                           1.6450 | −0.09 (−0.13 to −0.06) |  −0.24 (−0.31 to −0.16) |
| 1 min only                          | unseen containers, sparse |                           0.1533 | −6.37 (−8.28 to −4.76) |  −6.28 (−8.15 to −4.57) |
| 30 s, Aug to Oct                    | held-out time, dense      |                           1.5436 | −0.14 (−0.19 to −0.10) |  −0.32 (−0.40 to −0.25) |
| 30 s, Aug to Oct                    | held-out time, sparse     |                           0.1395 | −5.35 (−7.24 to −3.80) |  −6.26 (−9.96 to −3.73) |
| 30 s, Aug to Oct                    | unseen containers, dense  |                           1.7089 | −0.20 (−0.26 to −0.13) |  −0.32 (−0.41 to −0.24) |
| 30 s, Aug to Oct                    | unseen containers, sparse |                           0.1480 | −7.13 (−9.59 to −4.93) |  −7.13 (−9.73 to −4.93) |

All 72 cells are better with intervals clear of zero.
- **Dense.** The floor gains 0.09 to 0.42 %; on top of the average it adds up to 0.24 to 0.65 %.
- **Sparse.** The floor gains 1.1 to 8.1 %. On top of the average, seed 0 reaches 1.6 to 11.2 %.
  For seed 1 the average adds nothing on sparse tiles beyond what the floor already gives
  (−7.13 against −7.13 on the 30-second unseen containers): both fix the same band below zero.

**BOOMLET, paired per configuration**, one reference per table, its first row, with the
reference's geometric means over configurations; each cell below it is B's geometric mean, then
the percent change against the reference and the 95 % paired interval; negative is better. The
last column is the run's mean rank by CRPS among the 23 published models. The columns are:
- **90 MASE:** mean absolute scaled error of the median over the 90 scaled configurations, each
  divided by seasonal naive's.
- **90 CRPS:** continuous ranked probability score over the same 90.
- **10 s MASE:** the 45 configurations at 10-second width.
- **6 unscaled MAE:** mean absolute error of the median on the six low-variance configurations.

Floor on the final weights, resolution tag:

| Run                                        |                     90 MASE |                     90 CRPS |                   10 s MASE |              6 unscaled MAE | rank by CRPS |
| :----------------------------------------- | --------------------------: | --------------------------: | --------------------------: | --------------------------: | -----------: |
| final weights, resolution tag (reference)  |          0.8458 (reference) |          0.5219 (reference) |          0.9302 (reference) |          0.1557 (reference) |  12.57 of 24 |
| floor on the final weights, resolution tag | 0.8458, 0.00 (0.00 to 0.00) | 0.5219, 0.00 (0.00 to 0.00) | 0.9302, 0.00 (0.00 to 0.00) | 0.1557, 0.00 (0.00 to 0.00) |  12.57 of 24 |

Floor on the final weights, pick:

| Run                              |                     90 MASE |                     90 CRPS |                   10 s MASE |              6 unscaled MAE | rank by CRPS |
| :------------------------------- | --------------------------: | --------------------------: | --------------------------: | --------------------------: | -----------: |
| final weights, pick (reference)  |          0.8390 (reference) |          0.5176 (reference) |          0.9276 (reference) |          0.1546 (reference) |  12.36 of 24 |
| floor on the final weights, pick | 0.8390, 0.00 (0.00 to 0.00) | 0.5176, 0.00 (0.00 to 0.00) | 0.9276, 0.00 (0.00 to 0.00) | 0.1546, 0.00 (0.00 to 0.00) |  12.36 of 24 |

Floor on the 3-snapshot average, resolution tag:

| Run                                            |                     90 MASE |                     90 CRPS |                   10 s MASE |              6 unscaled MAE | rank by CRPS |
| :--------------------------------------------- | --------------------------: | --------------------------: | --------------------------: | --------------------------: | -----------: |
| 3-snapshot average, resolution tag (reference) |          0.8403 (reference) |          0.5192 (reference) |          0.9208 (reference) |          0.1525 (reference) |  12.42 of 24 |
| floor on the average, resolution tag           | 0.8403, 0.00 (0.00 to 0.00) | 0.5192, 0.00 (0.00 to 0.00) | 0.9208, 0.00 (0.00 to 0.00) | 0.1525, 0.00 (0.00 to 0.00) |  12.42 of 24 |

Floor on the 3-snapshot average, pick:

| Run                                  |                     90 MASE |                     90 CRPS |                   10 s MASE |              6 unscaled MAE | rank by CRPS |
| :----------------------------------- | --------------------------: | --------------------------: | --------------------------: | --------------------------: | -----------: |
| 3-snapshot average, pick (reference) |          0.8362 (reference) |          0.5164 (reference) |          0.9198 (reference) |          0.1510 (reference) |  12.30 of 24 |
| floor on the average, pick           | 0.8362, 0.00 (0.00 to 0.00) | 0.5164, 0.00 (0.00 to 0.00) | 0.9198, 0.00 (0.00 to 0.00) | 0.1510, 0.00 (0.00 to 0.00) |  12.30 of 24 |

The floor changed the scores of 4 of 96 configurations, all for the better:
- ds-189-10S short, CRPS −0.0005 %;
- ds-1520-10S short, medium and long, MAE −0.0002 to −0.0006 %.

In the other 38 configurations with eligible windows, the band was already non-negative.

The pairs against the thirds seed 0 run's 2026-11-02 copy show float noise below 1e-6 relative
in the other configurations; that run predates the float32 standardization of 2026-11-02. The
pairs against the plain average, run the same day as the floored runs, show none.

**BOOMLET leaderboard**, each run ranked alone against the 23 published models. **Mean CRPS
rank** is the average of the model's place by CRPS in each of the 90 configurations, and
**position** is the leaderboard's own sort by it.

| Model                                      | 90 MASE | 90 CRPS | 10 s MASE | 6 unscaled MAE | mean CRPS rank | position |
| :----------------------------------------- | ------: | ------: | --------: | -------------: | -------------: | -------: |
| final weights, resolution tag              |   0.692 |   0.623 |     0.716 |          0.156 |          12.57 | 12 of 24 |
| final weights + floor, resolution tag      |   0.692 |   0.623 |     0.716 |          0.156 |          12.57 | 12 of 24 |
| 3-snapshot average, resolution tag         |   0.688 |   0.620 |     0.709 |          0.152 |          12.42 | 12 of 24 |
| 3-snapshot average + floor, resolution tag |   0.688 |   0.620 |     0.709 |          0.152 |          12.42 | 12 of 24 |
| final weights, pick                        |   0.686 |   0.618 |     0.714 |          0.155 |          12.36 | 12 of 24 |
| final weights + floor, pick                |   0.686 |   0.618 |     0.714 |          0.155 |          12.36 | 12 of 24 |
| 3-snapshot average, pick                   |   0.684 |   0.616 |     0.708 |          0.151 |          12.30 | 12 of 24 |
| 3-snapshot average + floor, pick           |   0.684 |   0.616 |     0.708 |          0.151 |          12.30 | 12 of 24 |

## Reading

- **On our data, the floor is a free gain.** It is 0.1 to 0.4 % on busy metrics and 1 to 8 % on
  quiet ones, every cell resolved, and never worse by construction while the truth stays
  non-negative.
- **On BOOMLET it is inert, with no cost and no gain.** Only 3.5 % of its windows qualify, and on
  those the band was almost always non-negative already.
- **The case where it could cost has not been met in quantity.** That case is a series whose
  context is non-negative and whose future goes below zero. Our stores hold no negative values,
  and BOOMLET's eligible windows changed only 4 configurations, all for the better. A gauge that
  can cross zero (a temperature, a net flow, a delta) after a non-negative context is where to
  look if it ever shows.
