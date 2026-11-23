# Data: the three-width store and its draws

Since 2026-11-15 the release `aximoe-261115` trains on this data and more: [coarse widths](coarse-widths.md), [generated series](synthetic.md). What follows describes `aximoe-261105`.

Since 2026-11-23 the published model `aximoe-261123` trains on the same data as `aximoe-261115`.

In use since 2026-11-02, in the release `aximoe-261105`: its run (seed 2, since 2026-11-05)
trained on the store of record at a third of the draws per bin width.

## What it does

**Corpora and stores.** Container metrics of our own fleet at 10-second, 30-second and 1-minute
bins, each width over the same several months. A corpus is cut into tiles, the teacher's outputs
are stored beside them ([teacher](teacher.md)), and a merge links the 1-minute store, the two
30-second stores and the 10-second store into the store of record: its own
tile index, giving a series key or a channel name one code at every width, and a manifest naming
the sources, through which `store.Store.linked` and `Spanned` read each row.

**Tiles.** A tile is one series, one channel of one container, over
2,048 consecutive bins (`store.TILE`; 5.7 hours at 10 s, 17 hours at 30 s, 34 hours at 1 min)
starting at a multiple of 2,048 on the corpus grid, kept when at least 80 % of its values are
finite. A patch is 32 steps (`student.PATCH`), so a tile is 64 patches. Three channels are
dropped and 37 remain; each is **dense** (a value most of the time) or **sparse** (mostly zero)
by a fixed list. Each tile records its bin width (`width`).

**Splits** (in this order of precedence):
- **unseen channel:** every tile of `fs_write_bytes_rate`;
- **unseen containers:** every tile of a series key whose SHA-1 hash falls in the lowest 10 % of
  its range, so a container is held out at all three widths alike;
- **held-out time:** tiles starting on or after a split date two weeks before the data's end, of
  the remaining series;
- **train:** tiles ending before that date; the tile that straddles it is dropped.

**Training draws.** With `--width-shares 10:1,30:1,60:1`, `batches.Sampler` normalises the shares
to a third each; a draw takes one uniform number from the trainer's generator (`batches.Lcg`),
picks the width it falls in, then a training tile of that width uniformly, with replacement.

**Evaluation sample** (`batches.eval_tiles`). Up to 2,048 tiles of a split, half dense and half
sparse, shuffled by the generator at evaluation seed 0, every width in proportion to its tiles;
the unseen channel is sparse only, 1,024 tiles.

**Loader.** `batches.Prefetcher`, one thread four batches ahead, calls `batches.make_batch`:
`store.Store.rows` asks the kernel for every row of the batch over all four arrays at once
(`posix_fadvise` WILLNEED), then reads each row with `preadv`; `batches.standardise` builds the
inputs in float32 ([student](student.md)); the teacher's quantiles go to the card in float16.
The run trained at 75.0 steps/s on the A40 (an NVIDIA A40 card), standardising in float64;
through the float32 loader the model reproduces every tile score to four decimals.

## Why

Terms: **Winkler** is the interval score of the 0.1 to 0.9 band (width plus ten times the
distance of each point outside it) in the teacher's standardised space, lower is better.
**BOOMLET** is the 32-dataset subset of Datadog's BOOM observability benchmark, scored as
**MASE** (mean absolute scaled error of the median) and **CRPS** (continuous ranked probability
score) relative to seasonal naive over 90 scaled configurations. The **baseline** is the same
recipe on the 30-second and 1-minute store at 153,476 steps, seed 0;
the **thirds model** is the run of `aximoe-261103` before its tail average (the mean of its last
three snapshots).

- Mixed resolution: 30-second and 1-minute tiles together against 1-minute alone lowered
  BOOMLET MASE by 4.9 % (95 % interval −6.4 to −3.4), with the home 1-minute tiles unchanged
  within the single-run spread.
- Three and four widths: more widths were 0.2 to 0.4 % worse on dense cells over three pairs;
  the 10-second store of that test was much shorter than the others, which voids that answer.
- Three widths, equal exposure: the full-range 10-second corpus at 56 % of the draws gained 1.4
  to 3.1 % on 10-second tiles, cost 0.17 % (+0.0003 to +0.0045) and 0.43 % (+0.0052 to +0.0092)
  on the baseline store's dense tiles, and moved BOOMLET MASE by +0.61 % (0.9931 to 1.0209).
- Three widths, equal shares: at a third per width, level with the baseline on dense tiles
  (+0.06 %, −0.0011 to +0.0027 on held-out time), 1.84 % better on 10-second dense held-out
  time, BOOMLET MASE −0.88 % (0.9807 to 1.0020), sparse +0.97 % (+0.0006 to +0.0020); published
  2026-11-02 on the effect size.
- Exposure arm: at equal steps the two-width recipe trails the thirds model on BOOMLET by 1.05 %
  MASE (1.0007 to 1.0205) and by 1.4 to 2.3 % on every 10-second cell; the gain is the data, not
  the steps.
- Loader reads: the store's arrays exceed the page cache; faulting rows through
  a memory map trained at 16.3 steps/s, explicit reads with a float16 teacher at 75, above the
  baseline's 73.4.
- Width, routing and data scaling: on the 1-minute store, held-out time stays flat with half the
  training series (+0.01 %) and bends at about a quarter (+0.44 %), where 26 passes against 7.6
  make it repetition, not series count.

## Open

- Width shares between a third and the tile count.
- Breadth from data rather than the teacher, public observability corpora such as
  `Salesforce/cloudops_tsf`.
