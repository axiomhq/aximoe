# Recipe of the published model

The published model `aximoe-261123`, released 2026-11-23: the final weights of its run, seed 1.
Each step links the method page that says what is done. Against the model before it, `aximoe-261115`, it is 1.1 % better on BOOMLET's MASE
and CRPS with the channel tag picked per window (0.6218 against 0.6286, 0.5136 against 0.5192),
2.2 % and 2.0 % better on GIFT-Eval Web, and 0.5 to 0.6 % better by rollout on our own unseen
containers and unseen channels at the levels 0.1 to 0.9. The trade-off is again the upper outer
band: the levels 0.99 and above are 9 to 57 % worse on unseen channels and 16 to 32 % worse on
unseen containers. On BOOMLET, with the tag picked,
its mean rank by CRPS is 6.19 among the 23 published models and itself.

- **Data, fine widths:** 10-second, 30-second and 1-minute tiles of several months of our own fleet. [data](docs/methods/data.md)
- **Data, coarse widths:** the same fleet at 5 minutes, 15 minutes, 30 minutes and 1 hour. [coarse widths](docs/methods/coarse-widths.md)
- **Data, generated:** 495,202 stored generated training tiles with teacher forecasts, three draws in ten; 13 of every 64 tiles generated afresh, truth only. [generated series](docs/methods/synthetic.md)
- **Pulse trains:** half of the tiles generated afresh are baselines with events that repeat exactly, as timers and scheduled jobs draw them. [pulse trains](docs/methods/pulse-trains.md)
- **Late starts:** three in ten of the tiles generated afresh begin inside a patch after missing steps, and a position whose context holds no observed point has no loss. [late starts](docs/methods/late-starts.md)
- **Width shares:** of the draws on real tiles, 23.3 % to each fine width, 11.6 % to 5 minutes, 4.7 % to 15 minutes, 7 % each to 30 minutes and 1 hour. [coarse widths](docs/methods/coarse-widths.md)
- **Teacher:** Toto 2.0 313m's nine quantiles (levels 0.1 to 0.9) for every stored tile, computed once and stored beside it, at the store's boundaries and at the boundaries 16 steps after them. [teacher](docs/methods/teacher.md), [patch width](docs/methods/patch-width.md)
- **Far labels:** the teacher's forecasts of a tile's last patches made without seeing them, each read from step 0 and from step 16, learned at the unrolled positions. [far labels](docs/methods/far-labels.md)
- **Distillation:** a Huber distance from the student's nine shared quantiles to the teacher's, at a fixed weight of 1.0 beside the loss on the truth. [teacher](docs/methods/teacher.md)
- **Inputs:** 2,032 steps, 127 patches of 16, standardised with Toto 2.0's causal scaler and an asinh, the values embedded together with a presence mask. [patch width](docs/methods/patch-width.md), [student](docs/methods/student.md)
- **Band input:** beside each input step, its 0.1 and 0.9 levels as offsets from the value in the same asinh space, zero for an observed step, the forecast band for a fed-back one. [band input](docs/methods/band-input.md)
- **Backbone:** a dense causal transformer of four pre-norm layers, width 256, four attention heads and SwiGLU blocks of width 688. [student](docs/methods/student.md)
- **Positions:** every attention layer rotates its queries and keys by the patch position; the model has no learned position table. [rotary positions](docs/methods/rotary-positions.md)
- **Position mix:** each layer adds a learned mix of a position and the three before it, before the attention and before the block; 4,036,464 parameters in all. [position mix](docs/methods/position-mix.md)
- **Head:** fifteen quantile levels from 0.01 to 0.999 for each of the next 16 steps, monotone by construction. [student](docs/methods/student.md)
- **Loss on the truth:** pinball at all fifteen levels; the six outer levels learn from the truth alone. [student](docs/methods/student.md)
- **Unroll:** per step the last context patches, up to 17 on half of the steps and up to 61 on the other half, are replaced by the model's own forecasts, median and band, without gradients; every tile trains the rollout. [deep unroll](docs/methods/deep-unroll.md)
- **Frozen feedback:** fed-back patches are standardised with the statistics at the context's end, in training and in `forecast.py`; the MCP server's rollout standardises each fed-back patch with its own running statistics. [frozen feedback](docs/methods/frozen-feedback.md)
- **Channel tag:** one learned vector per channel of the tile index, plus an `unknown` row, added to every patch; the held-out channel's row is never trained, and its name maps to `unknown`. [tags](docs/methods/tags.md)
- **Resolution tag:** one learned vector per bin width, ten widths from 10 seconds to 1 hour, plus an `unknown` row, added the same way. [tags](docs/methods/tags.md)
- **Tag dropout:** each of the two tags becomes `unknown` with probability 0.1 per training tile, so the model also forecasts without them. [tags](docs/methods/tags.md)
- **Optimiser:** AdamW at a learning rate of 1.2e-3 after 200 warm-up steps, falling to zero along half a cosine, batch 64, gradients clipped at norm 1. [schedule](docs/methods/schedule.md), [training](docs/methods/training.md)
- **Length:** 1,175,000 steps; with 13 tiles of a batch generated afresh instead of 6, the stored tiles are drawn as often as for the model before it. [schedule](docs/methods/schedule.md)
- **Precision:** the passes in half precision, float16 with loss scaling; the teacher's quantiles stored in float16; the unroll's passes and the training pass replayed as CUDA graphs. [training](docs/methods/training.md)
- **Seed:** seed 1, the one run of this recipe with rotary positions. [schedule](docs/methods/schedule.md)
- **Weights:** the final weights. [schedule](docs/methods/schedule.md)
- **Floor:** in a window with no value below zero, every forecast quantile is clipped at zero. [floor](docs/methods/floor.md)
- **Rollout:** past the first 16 steps, each forecast patch's median joins the context with its 0.1 and 0.9 levels as its band, and the next patch is forecast from it, in `forecast.py` and in the MCP server alike. [rollout](docs/methods/rollout.md), [band input](docs/methods/band-input.md)
- **Frame:** with `forecast.py --frame context-at-floor`, which the published benchmark numbers use, a window whose scale at its last observed patch is at the floor keeps the given context's statistics for its fed-back patches. [frame](docs/methods/frame.md)
- **Tag at inference:** the channel's own row when its name is known, as in the MCP server; otherwise the row picked per window by the in-context fit of its own patches, which the published BOOMLET number uses, with the `unknown` row reported beside it. [tags](docs/methods/tags.md)
