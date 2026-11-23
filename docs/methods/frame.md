# The rollout's frame at the scale floor

In use since 2026-11-23 for the published benchmark numbers of
`data/models/aximoe-261123` (`forecast.py --frame context-at-floor`).

## What it does

Fed-back patches are standardised with a location and a scale ([frozen feedback](frozen-feedback.md)).
The forecaster's default takes them from the moved window's last observed patch. With
`--frame context-at-floor` a window whose own scale there is within a thousandth of the floor
(`batches.MIN_SCALE`, 1e-6) takes the given context's last-patch statistics instead; every other
window keeps the default.

## Why

A window that has turned constant has a scale at the floor, and a fed-back value slightly off the
constant then lies far outside that scale. On BOOMLET the frame changes nothing for the model
before this one (ratio 1.0000 on MASE and CRPS over the 90 scaled configurations) and repairs the
collapsed bands of models trained with late starts: CRPS 0.9871 (95 % paired interval 0.9681 to
1.0000) and 0.9814 (0.9453 to 1.0000) of the default frame with the channel tag unknown, 1.0000
with the tag picked.

## Limits

The MCP server's rollout keeps its own statistics and does not take this option.
