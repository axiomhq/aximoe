# Muon for the hidden matrices

Available since 2026-11-24 as `--optimizer muon`. The published model `aximoe-261123` was trained
with AdamW alone.

## What it does

With `--optimizer muon` the trainer runs two optimisers at one learning rate and one schedule.
Muon takes each layer's attention and feed-forward matrices, 20 tensors holding 3,162,112 of the
4,036,464 parameters; AdamW keeps the other 47 tensors.

For a matrix W with gradient G, Muon keeps a momentum M = 0.95 M + G, forms U = G + 0.95 M,
scales U to unit norm and runs five Newton-Schulz steps on it. The result has all its singular
values near 1: the update moves every direction of the matrix by about the same amount, where
the raw gradient moves a few directions far and the rest hardly at all. The matrix then changes
by the learning rate times `--muon-scale` (0.4) times the square root of its larger side, after
a weight decay of 0.01. The attention's input map holds the query, key and value maps stacked by
rows; each is treated as a matrix of its own.

Under half precision the loss scaler flags a step whose gradients are not finite. Both
optimisers then skip that step together (`muon.skip_together`). The resume file carries Muon's
momentum beside AdamW's state.

The method is from "Muon is Scalable for LLM Training" (Liu et al., arXiv 2502.16982).

## Why

It reaches the published model's scores in half the steps. With the published recipe otherwise
unchanged (16-step patches, far labels, rotary positions, seed 1, half precision), 587,500 steps
with Muon against the published model's 1,175,000 steps with AdamW, paired:

| measure (reference: `aximoe-261123`, AdamW, 1,175,000 steps) |  ratio | 95 % paired interval |
|--------------------------------------------------------------|-------:|---------------------:|
| `aximoe-261123` (reference), every row                       | 1.0000 |                      |
| BOOMLET MASE, channel tag picked                             | 0.9994 |     0.9958 to 1.0031 |
| BOOMLET CRPS, channel tag picked                             | 0.9991 |     0.9960 to 1.0021 |
| GIFT-Eval Web MASE                                           | 0.9908 |     0.9641 to 1.0116 |
| GIFT-Eval Web CRPS                                           | 1.0092 |     0.9928 to 1.0259 |

On BOOMLET's board the run takes the published model's place, 5 of 24, with a mean rank by CRPS
of 6.30 against 6.19. At equal length Muon is ahead: without rotary positions, 587,500 steps
with Muon read 0.9920 (0.9862 to 0.9971) on BOOMLET's MASE and 0.9908 (0.9858 to 0.9952) on its
CRPS against 587,500 steps with AdamW.

A step with Muon takes about a tenth longer than a step with AdamW, so half the steps are a
little over half the training time: 6 hours 16 minutes on an RTX 2080 Ti at 26.1 steps a second,
batch 64.

## Limits

One seed. On the six BOOMLET series scored without scaling, the median's absolute error is about
1 % higher with Muon in every run so far, while their CRPS is level. The scale was chosen among
0.1, 0.2 and 0.4 at 25,000 steps; at 587,500 steps, 0.4 and 0.1 are level on BOOMLET outside one
dataset.
