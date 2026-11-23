"""Serves the `container_metric_outliers` tool over MCP on stdio, or runs one investigation from the command
line:

    python python/aximoe/mcp_server.py                           # MCP server on stdin/stdout
    python python/aximoe/mcp_server.py investigate --dataset <dataset> --deployment <deployment> \\
        --containers '.*' --at 2026-10-25T17:45:00Z [--duration 60] [--threshold 0.99] [--history-days 1]

`AXIMOE_MODEL` is the model directory (`student.pt` and `model.json`, default
`<repo>/data/models/aximoe-261123`), at whose patch width the process runs; `AXIMOE_DEVICE` is
`cpu` (default) or `cuda`, `AXIMOE_MCP_LOG`
is the JSON-lines call log, and `AXIMOE_OUTPUT` is `text` (default) or `json`, which appends the
report as JSON. Diagnostics go to stderr, as stdout carries the protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from typing import Annotated

from pydantic import Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aximoe import investigate as inv  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DESCRIPTION = (
    "Outliers in Kubernetes container infrastructure metrics around a moment. Scans the containers matching "
    "`containers` (a regex) over a window centred on `at` (default 60 minutes), in 5-minute bins, for CPU use, CPU "
    "throttling, working-set memory and restarts, and flags every bin rarer than `threshold` (default 0.99) of the "
    "same series' bins over the previous `history_days` (default 1) days; it reads only these metrics and the OOM "
    "kills, even if the dataset holds others. "
    "Returns the containers with flagged bins, ranked by how many distinct metrics are flagged, then by the largest "
    "excess over the normal level: the top 10 with their time span, flagged metrics and largest excess (observed "
    "value and normal range), and a count of the rest. Every OOM kill in the window is listed by container and "
    "minute, unscored, and counts as one flagged metric in the ranking."
)

# when to use the tool and how to read it, shown to the agent with the server
INSTRUCTIONS = (
    "`container_metric_outliers` scores Kubernetes container infrastructure metrics (CPU, CPU throttling, "
    "working-set memory, restarts) and lists OOM kills. It reads only these, even if other metrics, like request "
    "rates, error rates or queue depths, exist in the dataset. If containers could be involved, "
    "call it early, alongside your first queries. Start with `.*`: the containers come ranked, most flagged metrics "
    "first, and a narrower pattern lists those beyond the top 10. Treat a container as OOM-killed only when an OOM "
    "kill is listed for it."
)


def model_directory() -> str:
    return os.environ.get("AXIMOE_MODEL", os.path.join(REPO, "data", "models", "aximoe-261123"))


def state() -> inv.State:
    model_dir = model_directory()
    device = os.environ.get("AXIMOE_DEVICE", "cpu")
    log = os.environ.get("AXIMOE_MCP_LOG", os.path.join(REPO, "data", "aximoe-mcp", "calls.jsonl"))
    output = os.environ.get("AXIMOE_OUTPUT", "text")
    if output not in ("text", "json"):
        sys.exit(f"aximoe-mcp: AXIMOE_OUTPUT is {output!r}; use json or text")
    try:
        model = inv.Model(model_dir, device)
        print(f"aximoe-mcp: loaded {model_dir} on {device}", file=sys.stderr)
        error = None
    except Exception as e:  # noqa: BLE001 - the server answers with robust z alone and says why
        model, error = None, str(e)
        print(f"aximoe-mcp: aximoe unavailable, robust z only: {e}", file=sys.stderr)
    return inv.State(model, error, log, output == "text")


async def run(st: inv.State, params: dict) -> tuple[dict | None, str | None]:
    started = dt.datetime.now(dt.timezone.utc)
    try:
        report = await inv.investigate(st, params["dataset"], params["containers"], params["at"],
                                       params.get("duration_minutes"), params.get("threshold"),
                                       params.get("deployment"), params.get("history_days"))
        error = None
    except Exception as e:  # noqa: BLE001 - every failure becomes the tool's error text
        report, error = None, str(e)
    inv.log_call(st.log, started, params, report, error)
    return report, error


def serve(st: inv.State) -> None:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("aximoe", instructions=INSTRUCTIONS)

    # plain text only: a structured copy reaches the agent as JSON-escaped `{"result": ...}`
    @server.tool(name="container_metric_outliers", description=DESCRIPTION, structured_output=False)
    async def investigate(
        dataset: Annotated[str, Field(description="MetricsDB dataset that holds the Kubernetes container "
                                                  "infrastructure metrics; the tool reads only those metrics "
                                                  "from it.")],
        deployment: Annotated[str, Field(description="Name of the Axiom deployment to read from, a table of "
                                                     "`~/.axiom.toml` on the server's machine.")],
        containers: Annotated[str, Field(description="Regular expression the container name must match; "
                                                     "`.*` for every container.")],
        at: Annotated[str, Field(description="The moment to look around, RFC 3339.")],
        duration_minutes: Annotated[int | None, Field(description="Length of the window centred on `at`; "
                                                                  "default 60, at most 240.")] = None,
        threshold: Annotated[float | None, Field(description="Share of the series' own history a bin must beat; "
                                                             "default 0.99.")] = None,
        history_days: Annotated[float | None, Field(description="Days of history each bin's rarity is measured "
                                                                "against; default 1, 0.25 to 14.")] = None,
    ) -> str:
        params = {"dataset": dataset, "containers": containers, "at": at, "duration_minutes": duration_minutes,
                  "threshold": threshold, "deployment": deployment, "history_days": history_days}
        report, error = await run(st, params)
        if error is not None:
            raise ValueError(error)
        out = inv.text(report)
        return out if st.text_only else out + "\n" + json.dumps(report)

    server.run("stdio")


def main(argv=None):
    if argv is None:
        # a run from the command line continues at the model's patch width
        inv.student.rerun_at_width(model_directory())
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "investigate":
        ap = argparse.ArgumentParser(prog="mcp_server.py investigate")
        ap.add_argument("--dataset", required=True)
        ap.add_argument("--containers", required=True)
        ap.add_argument("--at", required=True)
        ap.add_argument("--duration", type=int, default=None)
        ap.add_argument("--threshold", type=float, default=None)
        ap.add_argument("--deployment", required=True)
        ap.add_argument("--history-days", type=float, default=None)
        a = ap.parse_args(args[1:])
        st = state()
        params = {"dataset": a.dataset, "containers": a.containers, "at": a.at, "duration_minutes": a.duration,
                  "threshold": a.threshold, "deployment": a.deployment, "history_days": a.history_days}
        report, error = asyncio.run(run(st, params))
        if error is not None:
            sys.exit(f"error: {error}")
        print(inv.text(report))
        print(json.dumps(report, indent=2))
        return
    serve(state())


if __name__ == "__main__":
    main()
