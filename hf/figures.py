"""Draws the model card's figures into hf/images/:

    uv run python hf/figures.py [--model data/models/aximoe-261123] [--device cpu]

`forecast.png`: four of the example series, the last 96 steps held out, against the published
model's forecast of them. `eval.png`: the headline of the investigation eval of 11 and 12 Nov
2026, Claude Opus 5.5 agents on 31 Axiom production incidents with and
without Aximö's MCP server, as the change with Aximö and its 95 % interval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
from aximoe import forecast as fc  # noqa: E402
from aximoe import student  # noqa: E402

# Axiom's site colours; the band and highlight pair passes the dataviz palette checks on SURFACE
SURFACE = "#09101F"
INK = "#E6EAF2"
MUTED = "#8A94A6"
GRID = "#1C2436"
BAND = "#4A8FE8"
ACCENT = "#da5c2c"

HELD_OUT = 96
SHOWN_CONTEXT = 288
EXAMPLES = [("cpu_usage_rate", "CPU use", "cores"), ("mem_working_set", "Working-set memory", "GiB"),
            ("net_rx_bytes_rate", "Network received", "MB/s"), ("threads", "Threads", "")]
SCALE = {"GiB": 1 / 2**30, "MB/s": 1e-6, "cores": 1.0, "": 1.0}

# measure, without and with Aximö, change with Aximö and its 95 % interval, from the eval's report
EVAL = [("Time to the final report", "225 s", "205 s", -9.1, -14.5, -1.7),
        ("Input tokens read per run", "3.03 M", "2.45 M", -19.3, -25.1, -13.7),
        ("Output tokens per run", "17.1 k", "15.8 k", -7.4, -10.9, -4.0),
        ("Wrong reports", "36 of 370", "25 of 370", -30.6, -74.2, 0.0)]


def style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": [f for f in ("Inter", "Helvetica Neue", "DejaVu Sans") if f in installed], "font.size": 11,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "legend.frameon": False, "legend.labelcolor": INK,
    })


def fan(ax, x, levels: list[float], q: np.ndarray) -> None:
    """Nested quantile bands, the outermost faintest, and the median."""
    pairs = [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    for k, (lo, hi) in enumerate(pairs):
        ax.fill_between(x, q[levels.index(lo)], q[levels.index(hi)], color=BAND, alpha=0.14 + 0.08 * k, lw=0)
    ax.plot(x, q[levels.index(0.5)], color=BAND, lw=2)


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.955, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.895, subtitle, fontsize=12, color=MUTED, va="top")


def forecast_figure(description: dict, model, device, out: str) -> None:
    with open(os.path.join(REPO, "examples", "series.json")) as f:
        rows = json.load(f)["rows"]
    series = np.fromfile(os.path.join(REPO, "examples", "series.f32"), dtype="<f4").reshape(len(rows), -1)
    picked = [rows.index(name) for name, _, _ in EXAMPLES]
    context = series[picked, : series.shape[1] - HELD_OUT]
    windows = np.full((len(picked), fc.CONTEXT), np.nan, dtype=np.float32)
    windows[:, fc.CONTEXT - context.shape[1] :] = context
    tags = np.array([student.tag_id(description, name, 60) for name, _, _ in EXAMPLES], dtype=np.int64)
    q = fc.forecast(model, windows, tags, HELD_OUT, 64, device)
    levels = description["levels"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.8, bottom=0.08, hspace=0.42, wspace=0.14)
    header(fig, "Aximö forecasts the band, not just the line",
           "Four container metrics at 1-minute bins. The model reads the context and forecasts the next 96 minutes "
           "as 15 quantiles;\nthe held-out truth is drawn over the forecast.")
    n = series.shape[1]
    t = np.arange(-SHOWN_CONTEXT, HELD_OUT)
    for ax, i, (name, label, unit), qi in zip(axes.flat, picked, EXAMPLES, q):
        k = SCALE[unit]
        truth = series[i, n - HELD_OUT - SHOWN_CONTEXT :] * k
        fan(ax, t[SHOWN_CONTEXT:], levels, qi * k)
        ax.plot(t[: SHOWN_CONTEXT + 1], truth[: SHOWN_CONTEXT + 1], color=INK, lw=1.4)
        ax.plot(t[SHOWN_CONTEXT:], truth[SHOWN_CONTEXT:], color=ACCENT, lw=1.6)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (2, 3)))
        ax.set_title(f"{label}" + (f"  ·  {unit}" if unit else ""), loc="left", fontsize=13, pad=8)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([-288, -192, -96, 0, 96], ["−4.8 h", "−3.2 h", "−1.6 h", "now", "+1.6 h"])
    handles = [plt.Line2D([], [], color=INK, lw=1.4), plt.Line2D([], [], color=ACCENT, lw=1.6),
               plt.Line2D([], [], color=BAND, lw=2), plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.4, lw=0)]
    fig.legend(handles, ["context the model reads", "held-out truth", "forecast median",
                         "1–99 % · 5–95 % · 10–90 % · 20–80 % · 30–70 %"],
               loc="upper right", bbox_to_anchor=(0.97, 0.965), ncol=2, fontsize=10.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def eval_figure(out: str) -> None:
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes((0.25, 0.14, 0.5, 0.56))
    header(fig, "Agents with Aximö reach their report sooner on fewer tokens",
           "Claude Opus 5.5 agents investigating 31 Axiom production incidents, 12 rounds each, with the Axiom MCP "
           "server alone and with\nAximö's added. The change with Aximö and its 95 % interval, from resampling "
           "whole incidents; left of zero is better.")
    rows = np.arange(len(EVAL))[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    for y, (_, _, _, change, lo, hi) in zip(rows, EVAL):
        ax.plot([lo, hi], [y, y], color=ACCENT, lw=2.5, solid_capstyle="round")
        ax.scatter([change], [y], s=90, color=ACCENT, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.text(change, y + 0.22, f"{change:+.1f} %".replace("-", "−"), color=INK, fontsize=11.5, ha="center",
                va="bottom")
    ax.set_yticks(rows, [m for m, *_ in EVAL], fontsize=12.5, color=INK)
    ax.set_ylim(-0.6, len(EVAL) - 0.4)
    ax.set_xlim(-80, 10)
    ax.set_xticks([-75, -50, -25, 0], ["−75 %", "−50 %", "−25 %", "0"])
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("change with Aximö")
    fig.text(0.79, 0.73, "without", color=MUTED, fontsize=11, ha="center")
    fig.text(0.89, 0.73, "with Aximö", color=MUTED, fontsize=11, ha="center")
    for y, (_, without, with_, *_) in zip(rows, EVAL):
        yy = ax.transData.transform((0, y))[1]
        fy = fig.transFigure.inverted().transform((0, yy))[1]
        fig.text(0.79, fy, without, color=MUTED, fontsize=12, ha="center", va="center")
        fig.text(0.89, fy, with_, color=INK, fontsize=12, ha="center", va="center", fontweight="bold")
    fig.text(0.04, 0.035, "Time: mean over 334 runs per arm after dropping each arm's wrong reports and matching "
             "the counts. Tokens and wrong reports: all 370 runs per arm. Input tokens include cache reads.",
             color=MUTED, fontsize=9.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.path.join(REPO, "data", "models", "aximoe-261123"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "hf", "images"))
    args = ap.parse_args(argv)
    if argv is None:
        # a run from the command line continues at the model's patch width
        student.rerun_at_width(args.model)
    os.makedirs(args.out, exist_ok=True)
    style()
    device = torch.device(args.device)
    description, model = student.load(args.model, device)
    model.eval()
    with torch.no_grad():
        forecast_figure(description, model, device, os.path.join(args.out, "forecast.png"))
    eval_figure(os.path.join(args.out, "eval.png"))
    print(f"wrote forecast.png and eval.png to {args.out}")


if __name__ == "__main__":
    main()
