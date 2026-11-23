# Coarse bin widths

In use since 2026-11-15, in the release `aximoe-261115`.

Since 2026-11-23 the published model `aximoe-261123` trains on the same stores with the same width shares.

## What it does

The training store holds the fleet's series at seven bin widths: 10 seconds, 30 seconds and
1 minute, as before, and 5 minutes, 15 minutes, 30 minutes and 1 hour. A tile is 2,048 bins of
one series at one width, so a 1-hour tile spans 85 days where a 10-second tile spans under six
hours; the coarse tiles show the model daily and weekly cycles that never fit into a fine tile.

- **Sources.** The 5-minute and 15-minute tiles are cut from corpora fetched at those widths.
  The 30-minute and 1-hour tiles are the 15-minute corpus averaged over 2 and 4 consecutive
  bins; averaging stands in for a fetch: the 5-minute corpus averaged over 3 bins reproduces the
  fetched 15-minute corpus to a median relative difference of 1.3e-3 per channel or less.
- **Offsets.** A coarse series is short, so its tiles start at several offsets: five for 30
  minutes, nine for 1 hour.
- **Splits.** The same series, containers and channels are held out as in every store.
- **Draws.** Of the draws on real tiles, 23.3 % go to each fine width, 11.6 % to 5 minutes, 4.7 %
  to 15 minutes and 7 % each to 30 minutes and 1 hour
  (`--width-shares 10:1,30:1,60:1,300:0.5,900:0.2,1800:0.3,3600:0.3`).
- **Resolution tag.** The tag has a row for each of ten widths from 10 seconds to 1 hour and one
  for an unknown width ([tags](tags.md)).

## Why

BOOMLET, the public benchmark the model is judged on, holds series from 10 seconds to 1 hour; a
model trained at three fine widths had never seen the others. Together with the generated series
([synthetic](synthetic.md)) the coarse widths took 9.6 % off BOOMLET's MASE at 50,000 steps.
