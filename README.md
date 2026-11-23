# Aximö

Aximö is a time-series forecasting model, trained by Axiom on its own telemetry, for
OpenTelemetry and Kubernetes container metrics. For one metric series it reads the last 2,032
samples and returns the next 16 as a band of quantiles, from the 1st to the 99.9th percentile.
The band is what the rest of the stack uses: a value outside it is a surprise, and surprises
across the containers of a cluster are what an incident investigation looks for.

It learns from the series themselves, with the forecasts of Datadog's Toto 2.0 313m as a second
target for the inner band, the levels from 10 to 90 %. The model knows which
metric a series is (the channel tag: the cAdvisor and kube-state-metrics names every Kubernetes
cluster exports) and at which interval it was sampled (the resolution tag), and it carries an
`unknown` entry for both, so it can be run on metrics and intervals it has never seen.

## Status

The published model is `aximoe-261123`, 4,036,464 parameters. Its weights and its model card are
at [huggingface.co/axiomhq/aximoe](https://huggingface.co/axiomhq/aximoe);
[recipe.md](recipe.md) lists every step of its training.

BOOMLET is the smaller set of Datadog's public observability benchmark BOOM. The scores are
geometric means over its 90 configurations, lower is better: MASE, the mean absolute error of
the median divided by seasonal naive's, and CRPS, the continuous ranked probability score, which
scores the whole band. Aximö picks each window's channel tag from the window's own context
(`forecast.py --tag auto`), as it does on any series whose metric name it does not know. The
models are ordered as the benchmark ranks them, by mean rank by CRPS among the 23 published
models and Aximö; shown are the two above and the two below it:

| model            |   MASE |   CRPS | mean rank by CRPS |
|------------------|-------:|-------:|------------------:|
| Toto 2.0 313m    | 0.5853 | 0.4856 |              2.99 |
| Toto 2.0 22m     | 0.6008 | 0.4997 |              4.50 |
| `aximoe-261123`  | 0.6218 | 0.5136 |              6.19 |
| Toto 2.0 4m      | 0.6241 | 0.5172 |              6.38 |
| Toto 1.0         | 0.6166 | 0.5188 |              6.99 |

Forecasts further ahead than 16 samples feed the model its own output with its band. The model
card lists the limits: the levels from 99 % up, the time a forecast takes, and series that are
not container metrics.

## What is here

| path             | what                                                                                             |
|------------------|--------------------------------------------------------------------------------------------------|
| `python/aximoe/` | the model, its trainer and forecaster, the MCP (Model Context Protocol) server, the store format |
| `examples/`      | eight real series to run the forecaster on                                                       |
| `docs/methods/`  | a page per method in use                                                                         |
| `hf/`            | the model card on Hugging Face and its figures                                                   |
| `scripts/`       | publishing a trained model to Hugging Face                                                       |
| `data/`          | git-ignored; training stores and models, see Data below                                          |

## Getting started

```bash
uv sync
hf download axiomhq/aximoe --revision aximoe-261123 --local-dir data/models/aximoe-261123
uv run python python/aximoe/forecast.py --model data/models/aximoe-261123 \
    --in examples/series.f32 --out bands.f32 --resolution 60 --tag auto --length 2016
```

Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) and the `hf` command of the Hugging Face
CLI. The commands in the later sections run in the same environment: prefix them with `uv run`,
or activate `.venv` first. On a Mac, `--device mps` runs the model on the GPU; the trainer and
the server take the same value, and `--compile off` is the safe choice for the trainer there.
The example holds eight 1-minute series from our clusters, 2,016 samples each (`--length`; the
model's own window is 2,032 samples, and a shorter one counts as starting with missing samples).
`--resolution` and `--tag auto` give the model its two tags: the sampling interval, and for each
series the channel row that forecasts its own past best. The last command prints, per series,
the last observed value and the forecast median with its 10 to 90 % band at the first and the
last of 16 steps, and writes all 15 quantile levels of every step to `bands.f32`, laid out as
`bands.f32.json` says. The file formats and the other flags are in the script's docstring.

## Data

`data/` at the repository root (a directory or a symlink) holds these directories:

| directory              | holds                                                                      |
|------------------------|----------------------------------------------------------------------------|
| `data/derived/<name>/` | a training store: tiles with the teacher's forecasts for each              |
| `data/models/<name>/`  | a trained model: `student.pt` and `model.json`; experiment runs sit deeper |
| `data/aximoe-mcp/`     | the MCP server's call log                                                  |

A *tile* is one series over 2,048 consecutive samples; it is the unit of training and of the
stores. A store is five flat files (`values.f32`, `quantiles.f16`, `loc.f32`, `scale.f32`,
`manifest.json`) beside a tile index (`tiles.parquet`, `tiles.index.bin`, `tiles.index.json`),
readable from any language by mapping and seeking. A merged store holds only the index and a
manifest naming its source stores, and every read goes through to their arrays. Tiles are split into training, held-out
time, held-out containers and a held-out metric, so every number above is on data the model
never saw.

The tools that cut a corpus into tiles, run the teacher over them and merge stores are not part
of this repository; `python/aximoe/store.py` reads and writes the store's arrays.

## Training

The published model `aximoe-261123` trained on one merged store: several months of the
Kubernetes container metrics of Axiom's own fleet at seven bin widths from 10 seconds to 1 hour,
and stored generated series. All tiles carry Toto 2.0 313m's forecasts.

The store holds Axiom's own telemetry and is not published. In its directory sit the teacher's
forecasts on the 16-step grid (`shift16-313m/`, [patch width](docs/methods/patch-width.md)) and
its forecasts of hidden patches (`ahead-313m/` and `ahead16-313m/`,
[far labels](docs/methods/far-labels.md)). With such a store in place, the published model's run
was, from the repository root:

```bash
S=data/derived/<store>
A=$S/ahead-313m B=$S/ahead16-313m
AXIMOE_PATCH=16 AXIMOE_SHIFT_TEACHER=$PWD/$S/shift16-313m \
python python/aximoe/train.py --store $S/teacher-313m \
    --batch 64 --lr 1.2e-3 --warmup 200 --lr-decay cosine --steps 1175000 --eval-every 100000 \
    --eval-tiles 2048 --eval-seed 0 --eval-splits heldout_time,heldout_container,heldout_channel \
    --save-every 100000 --seed 1 \
    --width-shares 10:1,30:1,60:1,300:0.5,900:0.2,1800:0.3,3600:0.3 \
    --synthetic-share 0.3 --stream-share 0.2 --stream-family any-pulse --stream-late 0.3 \
    --empty-context-loss off --rope on --dtype fp16 \
    --canon 4 --unroll 17 --unroll-depths 17:0.5,61:0.5 --mask-tail 0 --frozen-feedback on \
    --ahead-teacher $A/coarse,$A/fine2,$A/fine3,$A/fine4,$A/fine5,$A/fine6,$A/fine,$A/synthetic,$B/coarse,$B/fine2,$B/fine3,$B/fine4,$B/fine5,$B/fine6,$B/fine,$B/synthetic \
    --save data/models/<run>/seed1
```

Training runs on one GPU
(`CUDA_VISIBLE_DEVICES` picks it): 16 hours 35 minutes on an RTX 2080 Ti, 19.7 steps a second.
Three draws in ten are a stored generated tile, the others pick a bin width by its share and
then a tile of that width, and 13 of every 64 tiles in a batch are generated afresh. The
learning rate falls to zero along half a cosine, and
the final weights are the published ones. Every `--resume-every` steps (default 5,000) the run
writes `resume.pt`, from which `--resume` continues it after a crash.
[recipe.md](recipe.md) lists every step of the recipe with its method page.

The run saves `student.pt` and, beside it, `model.json` with what running the weights needs:
the configuration, the quantile levels and the channel and resolution vocabulary; every
`--save-every` steps it also saves a snapshot, `student-<step>.pt`. Its standard output, kept as
`train.log`, holds the configuration and the scores on every split; `--load <dir> --steps 0`
scores a saved model. A run is named by its data and recipe, one directory per seed. The step
is compiled with `torch.compile`; `--compile off` runs it
eagerly. `--dtype` picks the passes' precision: half precision on a card by default (`bf16` where
the card computes it natively, `fp16` elsewhere), `f32` for single precision.

## The MCP server

One tool, `container_metric_outliers`, for an agent investigating an incident: which Kubernetes
containers' metrics behaved unusually around a moment? It needs an [Axiom](https://axiom.co)
organisation whose MetricsDB dataset holds the cAdvisor and kube-state-metrics series with a
`container` and `namespace` tag, and an API token. It reads its credentials from `~/.axiom.toml`
on the machine it runs on, the file the Axiom CLI uses. A call names a deployment, a table of
that file, and the server takes four fields from it:

```toml
[deployments.prod]
url    = "https://api.axiom.co"
edge   = "https://..."         # optional: your region's edge endpoint, where MPL queries go instead of url
token  = "xaat-..."            # an API token with query access
org_id = "your-org-id"         # the X-Axiom-Org-Id header
```

Every MetricsDB query is a POST to `<edge or url>/v1/query/_mpl` with that token and
organisation id; nothing else is consulted. A missing file, deployment or field comes back as
the call's error text, with the path in it.

```bash
python python/aximoe/mcp_server.py                     # the server on stdin/stdout
claude mcp add -s user aximoe -e AXIMOE_DEVICE=cpu -- $PWD/.venv/bin/python $PWD/python/aximoe/mcp_server.py   # cuda, or mps on a Mac
python python/aximoe/mcp_server.py investigate --dataset k8s-metrics-prod --containers '.*' \
    --at 2026-10-25T17:45:00Z --deployment prod [--duration 60] [--threshold 0.99] [--history-days 1]
```

| Variable         | Default                              | Meaning                                              |
|------------------|--------------------------------------|------------------------------------------------------|
| `AXIMOE_MODEL`   | `<repo>/data/models/aximoe-261123`   | the model directory: `student.pt` and `model.json`   |
| `AXIMOE_DEVICE`  | `cpu`                                | `cpu`, `cuda` (the first visible GPU) or `mps`       |
| `AXIMOE_MCP_LOG` | `<repo>/data/aximoe-mcp/calls.jsonl` | every call: time, parameters, report or error        |
| `AXIMOE_OUTPUT`  | `text`                               | `text`: the text report; `json`: the JSON after it   |

One call fetches, at 1-minute bins, four container metrics for the containers matching the
regex, summed over pods per namespace and container: CPU use, CPU throttling, working-set
memory and restarts, over the window plus its calibration history and 8 hours before it.
OOM kills are counted, not scored: a restart of a pod whose last termination reason is
`OOMKilled`, or a rise of `container_oom_events_total` with no such restart of the same pod
within 2 minutes (the counter misses kills that restart the container). The report lists them by container and
minute, and they count as one flagged metric in the ranking. It scans the window centred on `at` (default 60 minutes) in 5-minute bins with two
detectors: robust z, each point's distance from the median of the 4 hours before the window
in units of 1.4826 times the median absolute deviation; and the model, which forecasts the
window from the 8 hours before it one 16-minute patch at a time, each patch's median fed back
in, a point's score being how far it lies outside the forecast 10 to 90 % band, in band widths.
Each bin's score is calibrated against the same series' bins over the previous `history_days`
(default 1, 0.25 to 14), and every bin rarer than `threshold` for either detector is flagged.
The report ranks the containers with flagged bins by the number of distinct metrics flagged,
then by the largest excess over the normal level (how far the value lies beyond the range a
detector treats as normal, in multiples of the range's nearer edge), and lists the top 10 with
their time span, flagged metrics and largest excess; the rest are counted. The JSON report keeps
every flagged bin under `triggers`. A series with under half its points in the reference or the
context, or with no data in the 16 minutes before the window, which the model's forecast is read
from, is listed under `skipped`. A prompt that exercises it, once the
server is registered:

> Our checkout service returned errors from 17:45 UTC on 25 October. Use the dataset
> k8s-metrics-prod in the prod deployment and the container outlier tool to find which
> containers were involved.

The [model card](hf/README.md) reports how agents fared with and without the tool on 31 of our
incidents.

## How it works, briefly

The 2,032 samples are standardised against the series' own running mean and spread, cut into
127 blocks of 16, and embedded; the channel and resolution tags are added to every block as
learned vectors; four transformer layers let each block read the ones before it, by how far back
they lie; and a head on
the last block writes the 15 quantiles of the next block. Training scores every block's forecast
against the next block, both against the teacher's quantiles and against the truth, with the
outer quantile levels learned from the truth alone.

Findings that shaped it, each measured rather than assumed: the channel tag is worth 4 to 5 % on
quiet metrics and nothing on busy ones; a mixture-of-experts block does not beat a dense block
of the same active width at this data size; more parameters do not help; training on two
sampling intervals at once helps on every interval; the resolution tag changes nothing for a
single block ahead but halves the degradation of recursive forecasts on quiet metrics.

## Attribution

`python/aximoe/forecast.py` ports Toto 2.0's causal scaler, Apache-2.0, Copyright Datadog, Inc.

## License

Apache-2.0, see [LICENSE](LICENSE).
