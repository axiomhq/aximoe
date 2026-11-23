# Frozen statistics for fed-back patches

In use since 2026-11-15, in the release `aximoe-261115`
(`model.json` config: `frozen_feedback` true).

Since 2026-11-23 the published model `aximoe-261123` trains the same way and `forecast.py` forecasts so; its benchmark numbers use the forecaster's frame at the scale floor ([frame](frame.md)). The MCP server's rollout (`investigate.Model.forecast`) standardises each fed-back patch with the running statistics at its own end.

## What it does

Inputs are standardised with running statistics: patch `s` uses the location and scale at the
end of patch `s`. With `--frozen-feedback on`, every patch after the context's end, in training
and in forecasting alike, uses the statistics at the context's end, the last ones computed from
observed values.

## Why

The earlier trainer standardised a fed-back patch with the tile's own statistics at its position,
which include the truth the forecast replaces; the forecaster has no such values. The published
model trains in the space it forecasts in. At 50,000 steps the change was level on BOOMLET (MASE
ratio 1.0001, 95 % paired interval 0.9950 to 1.0052) and 0.8 % better on the far patches of our
own tiles.
