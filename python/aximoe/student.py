"""The student: a causal patch transformer of 4 layers at width 256, a channel tag and a resolution
tag added to every patch embedding, a feed-forward block that is dense or a mixture of experts,
and a quantile head whose fifteen knots are monotone by construction. Inputs are patches already
standardised with the teacher's scaler statistics, so student and teacher speak the same asinh
space. Tags can reach the backbone, added to every patch embedding, and the router, as an
additive bias on its logits; hard routing replaces the router by the tile's channel family.

A model directory holds `student.pt`, the weights, and `model.json`, everything a caller needs
to run them: the configuration, the quantile levels, the context, the patch width and the tag
vocabulary. A process runs at one patch width, `AXIMOE_PATCH` (32 steps by default); `forecast.py`
and `mcp_server.py` run from the command line take the width of the model they load
(`rerun_at_width`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

# steps per patch, from AXIMOE_PATCH; a tile of 2,048 steps holds SEQ patches
PATCH = int(os.environ.get("AXIMOE_PATCH", "32"))
if 2048 % PATCH:
    raise ValueError(f"AXIMOE_PATCH={PATCH} does not divide a tile of 2,048 steps")
SEQ = 2048 // PATCH
POSITIONS = SEQ - 1
CONTEXT = POSITIONS * PATCH
LEVELS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999)
# positions in LEVELS of the teacher's nine knots, 0.1 to 0.9
SHARED = (2, 3, 4, 5, 6, 7, 8, 9, 10)
MEDIAN = 6
# width of the guide input: a coarse forecast's median and two levels over a patch, a flag and the share of its context present
GUIDE = 3 * PATCH + 2
# a tag id holds the channel row below this and the resolution row plus one above it
TAG_BASE = 1 << 16
# channel families by name prefix; the last one collects everything else
FAMILIES = ("cpu", "mem", "psi", "fs", "other")


def family_of(channel: str) -> int:
    for i, prefix in enumerate(("cpu_", "mem_", "psi_", "fs_")):
        if channel.startswith(prefix):
            return i
    return len(FAMILIES) - 1


@dataclasses.dataclass
class StudentConfig:
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    d_ff: int = 688
    # rows of the channel table, one per channel plus `unknown`; zero removes the table
    tags: int = 0
    # the channel tag is added to every patch embedding
    backbone_tags: bool = False
    # experts per layer; zero keeps the dense feed-forward block
    experts: int = 0
    top_k: int = 1
    # the channel tag reaches the router as an additive bias on its logits
    router_tags: bool = False
    # the resolution row reaches the router the same way
    router_resolution: bool = False
    # rows of the resolution table, one per training bin width plus `unknown`; zero removes it
    resolutions: int = 0
    resolution_tags: bool = False
    # each tile's channel family picks the expert; no learned router
    hard_routing: bool = False
    # a flow-matching head beside the quantile head, sampled at inference
    flow_head: bool = False
    # patches the quantile head forecasts from each position, the next patch first
    horizon_patches: int = 1
    # each input step carries a 0.1 to 0.9 band as offsets from its value, zero for observed steps
    band_input: bool = False
    # trained with up to this many trailing input patches missing, so a position after the last
    # present patch forecasts further ahead and a horizon decodes in one pass; zero: next patch only
    mask_tail: int = 0
    # a masked training tile misses at least this many trailing patches, and a horizon decodes with as many missing
    mask_min: int = 0
    # each input position also holds a copy of the patch it forecasts, taken from the tile's own past, and its mask
    copy_input: bool = False
    # each input position also holds the mean of other series of the tile's group at its steps and the share of
    # this many series that back it; zero removes the input
    sibling_input: int = 0
    # each input position also holds the error of the model's own median forecast of its patch, made without this
    # input from the patches before it, and which steps have one; it takes the sibling input's place in every pass
    error_input: bool = False
    # with the error input, each of this many last observed positions also holds the error of a rollout that
    # started that many patches before the context end, and which steps have one; zero removes the input
    error_backtest: int = 0
    # the model keeps an embedding per series, read at every position like the channel tag: in each of this
    # many rounds it forecasts patches of the known context from a shorter prefix and writes the embedding
    # anew from that forecast beside the truth (`Student.update`); zero removes the embedding
    refine_rounds: int = 0
    # how many patches a round forecasts past its prefix
    refine_depth: int = 8
    # where the series embedding is read: "position", at every position after the round that made it; "forecast", from the context end on
    refine_read: str = "position"
    # each head attends twice, with the two halves of its queries and keys, and reads the first map minus a
    # learned share of the second (differential attention, Ye et al., arXiv 2410.05258)
    diff_attention: bool = False
    # the layers run this many times in sequence with the same weights
    loops: int = 1
    # hidden width of the patch embedding and of the head; -1 is four times d_model, 0 makes both one linear map
    io_hidden: int = -1
    # positions enter each attention as a rotation of queries and keys; the input carries no position embedding
    rope: bool = False
    # queries and keys are normalised per head and the queries scaled by a learned gain per head
    qk_norm: bool = False
    # each layer mixes this many positions, the position itself last, into its input and into its feed-forward input; zero removes the mix
    canon: int = 0
    # the mix reads only positions whose input patch holds a value
    canon_held: bool = False
    # state width of a recurrent layer run over the embedded patches before the first layer; zero removes it
    recurrent: int = 0
    # the last `across` layers attend, at each position, over the tiles of a tile's group instead of over its positions
    across: int = 0
    # a second head forecasts the level of the next patch, and a gate moves the patch's knots towards it
    level_head: bool = False
    # each input position also holds bin means of the series before it, 32 bins per scale of this many steps a bin
    history_scales: tuple = ()
    # the history input's values are held within this distance of zero; zero leaves them as they are
    history_clip: float = 0.0
    # the head adds learned weights of each position's last input value and input-patch median to its median
    anchors: bool = False
    # a forecasting position also holds the model's own forecast at a coarser bin width over its patch (`guide.py`):
    # per resolution row the steps per coarse bin and the coarse width's resolution row, (0, 0) for a width without one
    guide: tuple = ()
    # the patches a rollout feeds back and the forecasts after them are in the asinh space of the last observed patch
    frozen_feedback: bool = False

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "StudentConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        kept = {k: v for k, v in d.items() if k in fields}
        # a table with rows is in use unless the file says otherwise
        kept.setdefault("backbone_tags", kept.get("tags", 0) > 0)
        kept.setdefault("resolution_tags", kept.get("resolutions", 0) > 0)
        kept["history_scales"] = tuple(kept.get("history_scales", ()))
        kept["guide"] = tuple(tuple(int(v) for v in row) for row in kept.get("guide", ()))
        return cls(**kept)


def error_channel(pred: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """The error input `[n, SEQ, 2 * PATCH]` from a pass without it: position `s` holds the truth of
    its own patch minus the median `pred [n, SEQ, 15, PATCH]` forecast for it at position `s - 1`,
    held within 8 asinh units, and beside it which steps have a truth; zero at position 0 and
    after each tile's last observed position `known [n]`."""
    n, device = pred.shape[0], pred.device
    shown = (torch.arange(1, SEQ, device=device)[None] <= known[:, None])[:, :, None].to(pred.dtype)
    held = target_mask.to(pred.dtype) * shown
    error = ((targets.to(pred.dtype) - pred[:, :POSITIONS, MEDIAN]).clamp(-8.0, 8.0)) * held
    first = torch.zeros(n, 1, 2 * PATCH, device=device, dtype=pred.dtype)
    return torch.cat([first, torch.cat([error, held], dim=2)], dim=1)


def backtest_channel(fed: torch.Tensor, truth: torch.Tensor, present: torch.Tensor, cut: torch.Tensor, span: int) -> torch.Tensor:
    """The backtest half of the error input `[n, SEQ, 2 * PATCH]`: each of the `span` positions after
    `cut [n]` holds the truth of its patch minus the median a rollout from `cut` fed back there
    (`truth`, `fed [n, POSITIONS, PATCH]`, in one space), held within 8 asinh units, and beside it
    which steps have a truth (`present`); zero elsewhere."""
    n, device = fed.shape[0], fed.device
    at = torch.arange(POSITIONS, device=device)[None]
    shown = ((at > cut[:, None]) & (at <= cut[:, None] + span))[:, :, None].to(fed.dtype)
    held = present.to(fed.dtype) * shown
    error = (truth.to(fed.dtype) - fed).clamp(-8.0, 8.0) * held
    last = torch.zeros(n, 1, 2 * PATCH, device=device, dtype=fed.dtype)
    return torch.cat([torch.cat([error, held], dim=2), last], dim=1)


class MoeFfn(nn.Module):
    """Router plus `experts` SwiGLU blocks held as stacked weights. Each token goes to its `k`
    most probable experts, weighted by their router probability, or to `hard` when given."""

    def __init__(self, d_model: int, d_ff: int, experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, experts, bias=False)
        bound_in, bound_out = 1 / math.sqrt(d_model), 1 / math.sqrt(d_ff)
        self.w_in = nn.Parameter(torch.empty(experts, d_model, d_ff).uniform_(-bound_in, bound_in))
        self.v_in = nn.Parameter(torch.empty(experts, d_model, d_ff).uniform_(-bound_in, bound_in))
        self.b_in = nn.Parameter(torch.zeros(experts, 1, d_ff))
        self.b_v = nn.Parameter(torch.zeros(experts, 1, d_ff))
        self.w_out = nn.Parameter(torch.empty(experts, d_ff, d_model).uniform_(-bound_out, bound_out))
        self.b_out = nn.Parameter(torch.zeros(experts, 1, d_model))

    @property
    def experts(self) -> int:
        return self.w_in.shape[0]

    def expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        # the biases join in the product's dtype, so the block runs under bf16 autocast
        a, b = x @ self.w_in[e], x @ self.v_in[e]
        h = F.silu(a + self.b_in[e].to(a.dtype)) * (b + self.b_v[e].to(b.dtype))
        o = h @ self.w_out[e]
        return o + self.b_out[e].to(o.dtype)

    def forward(self, x: torch.Tensor, k: int, router_bias: torch.Tensor | None, hard: torch.Tensor | None):
        """`x`: `[tokens, d_model]`. Returns the mixed output, the Switch balance loss
        `E * sum_e(share of items routed to e * mean router probability of e)`, zero under hard
        routing, and the experts each token chose, `[tokens, k]`."""
        tokens = x.shape[0]
        if hard is not None:
            gates, chosen, probs = torch.ones(tokens, 1, device=x.device, dtype=x.dtype), hard[:, None], None
        else:
            logits = self.router(x)
            if router_bias is not None:
                logits = logits + router_bias
            probs = F.softmax(logits, dim=1)
            if k == 1:
                chosen = probs.argmax(1, keepdim=True)
                gates = probs.gather(1, chosen)
            else:
                gates, chosen = probs.topk(k, dim=1)
        y = torch.zeros_like(x)
        for e in range(self.experts):
            mask = chosen == e
            ids = mask.any(1).nonzero().squeeze(1)
            if ids.numel() == 0:
                continue
            gate = (gates * mask).sum(1)[ids]
            y = y.index_add(0, ids, self.expert(x[ids], e) * gate[:, None])
        if probs is None:
            balance = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            counts = torch.bincount(chosen.flatten(), minlength=self.experts).to(x.dtype)
            balance = (counts / chosen.numel() * probs.mean(0)).sum() * self.experts
        return y, balance, chosen


def rotate(x: torch.Tensor, positions: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
    """`x [n, heads, s, width]` with every pair of dimensions turned by its position times the
    pair's rate `turn [width / 2]`; `positions`: `[s]`, or `[n, s]` per tile."""
    angle = positions.to(x.dtype)[..., None] * turn
    angle = angle[None, None] if angle.dim() == 2 else angle[:, None]
    cos, sin = angle.cos(), angle.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


def mix_back(x: torch.Tensor, weight: torch.Tensor, held: torch.Tensor | None = None) -> torch.Tensor:
    """`x [n, s, d]` plus, per dimension, the sum of each position and the positions before it
    under `weight [d, 1, k]`, whose last column weighs the position itself; with `held [n, s, 1]`,
    one or zero, the sum reads the held positions only."""
    k = weight.shape[2]
    source = x if held is None else x * held
    return x + F.conv1d(F.pad(source.transpose(1, 2), (k - 1, 0)), weight, groups=weight.shape[0]).transpose(1, 2)


def mix_back_at(kept: torch.Tensor, p: torch.Tensor, new: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """What `mix_back` adds at position `p [n]` alone, `[n, 1, d]`, when `new [n, d]` stands at
    `p` and `kept [n, s, d]` holds the positions before it."""
    k, d = weight.shape[2], kept.shape[2]
    at = p[:, None] + torch.arange(1 - k, 0, device=p.device)
    before = kept.gather(1, at.clamp_min(0)[:, :, None].expand(-1, -1, d)) * (at >= 0)[:, :, None]
    w = weight[:, 0].T
    return (before * w[:-1]).sum(1, keepdim=True) + new[:, None] * w[-1:]


class Layer(nn.Module):
    """Pre-norm layer: causal self-attention, then a SwiGLU block or the MoE layer."""

    def __init__(self, cfg: StudentConfig, across: bool = False, depth: int = 0):
        super().__init__()
        self.across = across
        self.diff = cfg.diff_attention and not across
        if self.diff:
            if cfg.rope or cfg.qk_norm:
                raise ValueError("differential attention is built without rotary positions and normalised queries and keys")
            half = cfg.d_model // cfg.heads // 2
            # the share's start grows with the layer, as in the paper
            self.share_start = 0.8 - 0.6 * math.exp(-0.3 * depth)
            self.share_q = nn.Parameter(torch.randn(2, half) * 0.1)
            self.share_k = nn.Parameter(torch.randn(2, half) * 0.1)
            self.head_norm = nn.RMSNorm(cfg.d_model // cfg.heads, eps=1e-5)
        # zero weights start the layer as one without the mix
        self.canon_a = nn.Parameter(torch.zeros(cfg.d_model, 1, cfg.canon)) if cfg.canon else None
        self.canon_c = nn.Parameter(torch.zeros(cfg.d_model, 1, cfg.canon)) if cfg.canon else None
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-4)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, bias=True, batch_first=True)
        self.own_attention = cfg.rope or cfg.qk_norm or self.diff
        half = cfg.d_model // cfg.heads // 2
        self.register_buffer("turn", 10000.0 ** (-torch.arange(half) / half) if cfg.rope and not across else None,
                             persistent=False)
        self.qk_gain = nn.Parameter(torch.ones(cfg.heads, 1, 1)) if cfg.qk_norm else None
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-4)
        if cfg.experts:
            self.moe = MoeFfn(cfg.d_model, cfg.d_ff, cfg.experts)
        else:
            self.moe = None
            self.w_in = nn.Linear(cfg.d_model, cfg.d_ff)
            self.v_in = nn.Linear(cfg.d_model, cfg.d_ff)
            self.w_out = nn.Linear(cfg.d_ff, cfg.d_model)

    def attend(self, queries: torch.Tensor, context: torch.Tensor, blocked: torch.Tensor, q_at: torch.Tensor,
               k_at: torch.Tensor) -> torch.Tensor:
        """This layer's attention under the configuration's rotary positions and normalised
        queries and keys: `queries [n, a, d]` attend to `context [n, b, d]` wherever `blocked`
        (broadcast to `[n, heads, a, b]`) is false; `q_at`, `k_at` are their positions, `[a]` and
        `[b]` or per tile `[n, a]`."""
        n, a, d = queries.shape
        heads = self.attn.num_heads
        w, bias = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(queries, w[:d], bias[:d]).view(n, a, heads, d // heads).transpose(1, 2)
        k = F.linear(context, w[d : 2 * d], bias[d : 2 * d]).view(n, -1, heads, d // heads).transpose(1, 2)
        v = F.linear(context, w[2 * d :], bias[2 * d :]).view(n, -1, heads, d // heads).transpose(1, 2)
        if self.qk_gain is not None:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * self.qk_gain.to(q.dtype)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.turn is not None:
            q, k = rotate(q, q_at, self.turn), rotate(k, k_at, self.turn)
        if self.diff:
            half = q.shape[-1] // 2
            first = torch.softmax((q[..., :half] @ k[..., :half].transpose(-1, -2) * half ** -0.5).masked_fill(blocked, float("-inf")), dim=-1)
            second = torch.softmax((q[..., half:] @ k[..., half:].transpose(-1, -2) * half ** -0.5).masked_fill(blocked, float("-inf")), dim=-1)
            out = self.head_norm((first - self.share().to(first.dtype) * second) @ v) * (1.0 - self.share_start)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)
        return self.attn.out_proj(out.transpose(1, 2).reshape(n, a, d))

    def share(self) -> torch.Tensor:
        """The share of the second attention map that differential attention takes off the first."""
        return (torch.exp((self.share_q[0] * self.share_k[0]).sum()) - torch.exp((self.share_q[1] * self.share_k[1]).sum())
                + self.share_start)

    def attend_across(self, h: torch.Tensor, reads: torch.Tensor | None) -> torch.Tensor:
        """This layer's attention over tiles: at each position, tile `i` of `h [n, s, d]` reads
        the tiles `j` with `reads [s, n, n]` true at `[position, i, j]`; without `reads`, every
        tile reads itself alone."""
        n, s, d = h.shape
        heads = self.attn.num_heads
        w, bias = self.attn.in_proj_weight, self.attn.in_proj_bias
        v = F.linear(h, w[2 * d :], bias[2 * d :])
        if reads is None:
            return self.attn.out_proj(v)
        q = F.linear(h, w[:d], bias[:d]).view(n, s, heads, d // heads).permute(1, 2, 0, 3)
        k = F.linear(h, w[d : 2 * d], bias[d : 2 * d]).view(n, s, heads, d // heads).permute(1, 2, 0, 3)
        if self.qk_gain is not None:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * self.qk_gain.to(q.dtype)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        out = F.scaled_dot_product_attention(q, k, v.view(n, s, heads, d // heads).permute(1, 2, 0, 3), attn_mask=reads[:, None])
        return self.attn.out_proj(out.permute(2, 0, 1, 3).reshape(n, s, d))

    def keys_values(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys and values this layer's attention reads of `h [n, s, d]` at `positions`
        (`[s]`, or `[n, s]` per tile), side by side: `[n, s, 2 * d]`."""
        n, s, d = h.shape
        kv = F.linear(h, self.attn.in_proj_weight[d:], self.attn.in_proj_bias[d:])
        if not self.own_attention:
            return kv
        k = kv[..., :d].view(n, s, self.attn.num_heads, -1).transpose(1, 2)
        if self.qk_gain is not None:
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.turn is not None:
            k = rotate(k, positions, self.turn)
        return torch.cat([k.transpose(1, 2).reshape(n, s, d), kv[..., d:]], dim=2)

    def attend_cached(self, h: torch.Tensor, kept: torch.Tensor, kv: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """This layer's attention of `h [n, 1, d]` at position `p [n]` over the keys and values
        `kept [n, S, 2 * d]` of the positions before `p` and its own `kv [n, 1, 2 * d]`."""
        n, S, d = kept.shape[0], kept.shape[1], h.shape[2]
        heads = self.attn.num_heads
        w, bias = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(h, w[:d], bias[:d]).view(n, 1, heads, -1).transpose(1, 2)
        if self.qk_gain is not None:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * self.qk_gain.to(q.dtype)
        if self.turn is not None:
            q = rotate(q, p[:, None], self.turn)
        q = q[:, :, 0]
        keys, values = kept[..., :d].view(n, S, heads, -1), kept[..., d:].view(n, S, heads, -1)
        own_k, own_v = kv[:, 0, :d].view(n, heads, -1), kv[:, 0, d:].view(n, heads, -1)
        later = (torch.arange(S, device=p.device)[None] >= p[:, None])[:, :, None]

        def read(part: slice) -> torch.Tensor:
            # a softmax over the cached positions and the own one, kept apart so the cache is read in place
            scale = q[..., part].shape[-1] ** -0.5
            cached = ((keys[..., part] * q[:, None, :, part]).sum(-1) * scale).masked_fill(later, float("-inf"))
            own = (own_k[..., part] * q[..., part]).sum(-1) * scale
            top = torch.maximum(cached.amax(1), own)
            e_cached, e_own = torch.exp(cached - top[:, None]), torch.exp(own - top)
            total = e_cached.sum(1) + e_own
            return (e_cached[..., None] * values).sum(1) / total[..., None] + (e_own / total)[..., None] * own_v

        if self.diff:
            half = q.shape[-1] // 2
            out = self.head_norm(read(slice(0, half)) - self.share().to(q.dtype) * read(slice(half, None))) * (1.0 - self.share_start)
        else:
            out = read(slice(None))
        return self.attn.out_proj(out.reshape(n, 1, d))

    def forward(self, x, causal, k, router_bias, hard, positions=None, kept=None, reads=None, held=None):
        """The layer's output; `kept`, a list, receives what `Student.next_position` reads of
        earlier positions: the attention's keys and values (`keys_values`), under `canon`
        followed along the last axis by what the two mixes read, the layer's input and its state
        after the attention, each times `held` when given. A layer across tiles attends under
        `reads` (`attend_across`)."""
        before = x if held is None else x * held
        if self.canon_a is not None:
            x = mix_back(x, self.canon_a, held)
        h = self.norm1(x)
        if self.across:
            x = x + self.attend_across(h, reads)
        elif self.own_attention:
            x = x + self.attend(h, h, causal, positions, positions)
        else:
            x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        if kept is not None:
            kv = self.keys_values(h, positions)
            kept.append(kv if self.canon_a is None else torch.cat([kv, before, x if held is None else x * held], dim=2))
        if self.canon_c is not None:
            x = mix_back(x, self.canon_c, held)
        h = self.norm2(x)
        if self.moe is None:
            return x + self.w_out(F.silu(self.w_in(h)) * self.v_in(h)), None
        n, s, d = h.shape
        y, balance, chosen = self.moe(h.reshape(n * s, d), k, router_bias, hard)
        return x + y.reshape(n, s, d), (balance, chosen)


def monotone(raw: torch.Tensor) -> torch.Tensor:
    """Knots from raw head outputs `[n, SEQ, 15, PATCH]`: the median is taken as is, every knot
    above it adds a positive increment to its lower neighbour, every knot below subtracts one
    from its upper neighbour, so the fifteen levels never cross."""
    median = raw[:, :, MEDIAN : MEDIAN + 1]
    down = F.softplus(raw[:, :, :MEDIAN])
    up = F.softplus(raw[:, :, MEDIAN + 1 :])
    below = median - down.flip(2).cumsum(2).flip(2)
    above = median + up.cumsum(2)
    return torch.cat([below, median, above], dim=2)


def patch_level(path: torch.Tensor) -> torch.Tensor:
    """The level of each patch of `path [..., PATCH]`: the median over its steps. Its gradient
    is the mean's, so a change of the level moves every step of the patch alike."""
    mean = path.mean(dim=-1)
    return mean + (path.median(dim=-1).values - mean).detach()


def knot_draws(knots: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Values of the quantile functions `knots [..., 15, steps]` at the probabilities `u [...,
    samples, steps]`, linear between the two levels around each probability; outside the
    levels the end knots hold."""
    levels = torch.tensor(LEVELS, device=knots.device, dtype=knots.dtype)
    k = knots.transpose(-1, -2).unsqueeze(-3)
    x = k[..., 0].expand_as(u).clone()
    for m in range(len(LEVELS) - 1):
        w = ((u - levels[m]) / (levels[m + 1] - levels[m])).clamp(0.0, 1.0)
        x = x + w * (k[..., m + 1] - k[..., m])
    return x


def anchor_values(inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The last present value and the median of the present values of each position's input
    patch, `[n, s]` each for `inputs`, `mask` `[n, s, PATCH]`; a position whose patch holds no
    value carries the pair of the last position that does, and zero stands before the first."""
    present = mask > 0
    s, steps = inputs.shape[1], inputs.shape[2]
    last_at = (present * torch.arange(1, steps + 1, device=inputs.device)).amax(dim=2)
    last = inputs.gather(2, (last_at - 1).clamp_min(0)[..., None])[..., 0]
    median = torch.nanmedian(inputs.masked_fill(~present, float("nan")), dim=2).values
    source = torch.where(last_at > 0, torch.arange(s, device=inputs.device)[None], -1).cummax(dim=1).values
    seen = source >= 0
    source = source.clamp_min(0)
    zero = torch.zeros((), dtype=inputs.dtype, device=inputs.device)
    return (torch.where(seen, last.gather(1, source), zero),
            torch.where(seen, torch.nan_to_num(median).gather(1, source), zero))


class FlowHead(nn.Module):
    """The velocity field of a conditional flow from standard normal noise to the next patch:
    a two-layer MLP over the backbone state, the noisy patch and a sinusoidal time embedding.
    The state enters through its own first-layer projection, so one projection per position
    serves every sample and step."""

    def __init__(self, d_model: int, hidden: int, bands: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(0.0, 3.0, bands), persistent=False)
        self.state = nn.Linear(d_model, hidden)
        self.noisy = nn.Linear(PATCH + 2 * bands, hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, PATCH)

    def forward(self, state: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """`state`: the projected backbone state `[..., hidden]`; `x`: the noisy patch
        `[..., PATCH]`; `t`: the flow time in [0, 1], `[..., 1]`; broadcast together."""
        phase = t * self.freqs
        embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).expand(*x.shape[:-1], -1)
        z = F.silu(state + self.noisy(torch.cat([x, embedding], dim=-1)))
        return self.out(F.silu(self.mid(z)))


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        if (cfg.backbone_tags or cfg.router_tags) and cfg.tags == 0:
            raise ValueError("tags need table rows")
        if (cfg.router_tags or cfg.router_resolution or cfg.hard_routing) and cfg.experts == 0:
            raise ValueError("routing needs experts")
        if cfg.router_resolution and cfg.resolutions == 0:
            raise ValueError("a resolution router bias needs resolution rows")
        if cfg.hard_routing and cfg.experts != len(FAMILIES):
            raise ValueError("hard routing needs one expert per family")
        if cfg.resolution_tags and cfg.resolutions == 0:
            raise ValueError("resolution tags need table rows")
        if cfg.anchors and (cfg.flow_head or cfg.horizon_patches > 1):
            raise ValueError("anchors need a quantile head of one patch")
        if cfg.across and cfg.experts:
            raise ValueError("layers across tiles need a dense student")
        if cfg.level_head and (cfg.flow_head or cfg.horizon_patches > 1):
            raise ValueError("the level head needs a quantile head of one patch")
        self.cfg = cfg
        if cfg.error_input and cfg.sibling_input:
            raise ValueError("the error input takes the sibling input's place")
        if cfg.error_backtest and not cfg.error_input:
            raise ValueError("the backtest errors need the error input")
        if cfg.refine_rounds and (cfg.sibling_input or cfg.error_input or cfg.experts):
            raise ValueError("the series embedding needs a dense student without sibling or error input")
        self.side = (2 * (cfg.sibling_input > 0 or cfg.error_input or cfg.refine_rounds > 0) + 2 * (cfg.error_backtest > 0)) * PATCH
        fed = (2 + 2 * cfg.band_input + 2 * cfg.copy_input) * PATCH + self.side
        wide = 4 * cfg.d_model if cfg.io_hidden < 0 else cfg.io_hidden
        self.embed1 = nn.Linear(fed, wide or cfg.d_model)
        self.embed2 = nn.Linear(wide, cfg.d_model) if wide else None
        side = len(cfg.history_scales) * 2 * PATCH + (GUIDE if cfg.guide else 0)
        self.history_in = nn.Linear(side, cfg.d_model, bias=False) if side else None
        if self.history_in is not None:
            # zero weights start the model as one without the history input and the guide
            nn.init.zeros_(self.history_in.weight)
        self.pos = None if cfg.rope else nn.Embedding(SEQ, cfg.d_model)
        self.recurrent = nn.GRU(cfg.d_model, cfg.recurrent, batch_first=True) if cfg.recurrent else None
        self.recurrent_out = nn.Linear(cfg.recurrent, cfg.d_model) if cfg.recurrent else None
        if self.recurrent_out is not None:
            # zero weights start the model as one without the recurrent layer
            nn.init.zeros_(self.recurrent_out.weight)
            nn.init.zeros_(self.recurrent_out.bias)
        self.tags = nn.Embedding(cfg.tags, cfg.d_model) if cfg.backbone_tags else None
        self.series_start = nn.Parameter(torch.zeros(cfg.d_model)) if cfg.refine_rounds else None
        self.series_out = nn.Linear(cfg.d_model, cfg.d_model) if cfg.refine_rounds else None
        if self.series_out is not None:
            # zero weights leave the embedding at its start until trained
            nn.init.zeros_(self.series_out.weight)
            nn.init.zeros_(self.series_out.bias)
        self.router_tags = nn.Embedding(cfg.tags, cfg.experts) if cfg.router_tags else None
        self.router_resolution = nn.Embedding(cfg.resolutions, cfg.experts) if cfg.router_resolution else None
        self.resolution_tags = nn.Embedding(cfg.resolutions, cfg.d_model) if cfg.resolution_tags else None
        self.layers = nn.ModuleList(Layer(cfg, i >= cfg.layers - cfg.across, i) for i in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-4)
        self.head1 = nn.Linear(cfg.d_model, wide) if wide else None
        self.head2 = nn.Linear(wide or cfg.d_model, len(LEVELS) * PATCH * cfg.horizon_patches)
        self.anchor_head = nn.Linear(wide or cfg.d_model, PATCH + 1) if cfg.anchors else None
        if self.anchor_head is not None:
            # zero weights start the model as one without anchors
            nn.init.zeros_(self.anchor_head.weight)
            nn.init.zeros_(self.anchor_head.bias)
        self.level_head = nn.Linear(wide or cfg.d_model, 2) if cfg.level_head else None
        if self.level_head is not None:
            # a zero level and a nearly closed gate start the model as one without the level head
            nn.init.zeros_(self.level_head.weight)
            with torch.no_grad():
                self.level_head.bias.copy_(torch.tensor([0.0, -4.0]))
        self.flow = FlowHead(cfg.d_model, 2 * cfg.d_model) if cfg.flow_head else None
        self.register_buffer("causal", torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1),
                             persistent=False)
        self.register_buffer("positions", torch.arange(SEQ), persistent=False)

    def reads(self, group: torch.Tensor | None, mask: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor | None:
        """What a layer across tiles lets each tile read, `[s, n, n]`: itself, and each tile of
        its `group [n]` from that tile's first present patch on (`mask [n, s, PATCH]`). With `p
        [n]`, one position per tile: `[1, n, n]`, each tile at its own position, reading the
        tiles of its group that stand at the same one. Without `group`, nothing: every tile
        reads itself alone."""
        if group is None or not self.cfg.across:
            return None
        n = mask.shape[0]
        started = (mask.sum(2) > 0).cummax(dim=1).values
        same = group[:, None] == group[None, :]
        own = torch.eye(n, dtype=torch.bool, device=mask.device)
        if p is None:
            return (same[None] & started.T[:, None, :]) | own[None]
        here = started.gather(1, p[:, None])[:, 0]
        return ((same & (p[:, None] == p[None, :]) & here[None, :]) | own)[None]

    def forward_full(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None = None,
                     families: torch.Tensor | None = None, band: torch.Tensor | None = None,
                     copy: torch.Tensor | None = None, sibling: torch.Tensor | None = None,
                     group: torch.Tensor | None = None, history: torch.Tensor | None = None,
                     state: torch.Tensor | None = None):
        """`inputs`, `mask`: `[n, SEQ, PATCH]`; `tags`: one packed id per tile; `families`: one
        family index per tile, needed under hard routing; `band`: under `band_input`, the 0.1
        and 0.9 levels' offsets from each input step, `[n, SEQ, 2 * PATCH]`, zero when absent;
        `copy`: under `copy_input`, `periodic.copy_ahead`'s `[n, SEQ, 2 * PATCH]`, zero when absent;
        `sibling`: under `sibling_input`, `batches.sibling_channel`'s `[n, SEQ, 2 * PATCH]`, zero when absent;
        `group`: under `across`, one group number per tile, the tiles of a group covering the same
        steps; without it every tile is alone; `history`: under `history_scales`,
        `batches.history_channel`'s `[n, SEQ, scales * 2 * PATCH]`, under `guide`, `guide.channel`'s
        `[n, SEQ, GUIDE]`, under both the two side by side in that order; nothing when absent.
        Returns quantiles `[n, SEQ, 15, PATCH]`, monotone over the knot axis, where position `s`
        predicts patch `s + 1`, the mean balance loss over the MoE layers (zero for a dense
        student) and the experts each token chose in every MoE layer, `[n * SEQ, k]` per layer."""
        h, balance, chosen = self.hidden(inputs, mask, tags, families, band, copy, sibling, group, history, state)
        return self.head(h, anchor_values(inputs, mask) if self.cfg.anchors else None), balance, chosen

    def head(self, h: torch.Tensor, anchors: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """The quantile head's knots for the next patch, `[n, SEQ, 15, PATCH]`, of the normalised
        state `h`; `anchors` as `head_patches` takes them."""
        return self.head_patches(h, anchors)[..., :PATCH]

    def head_patches(self, h: torch.Tensor, anchors: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """The quantile head's knots for every patch it forecasts, `[n, positions, 15,
        horizon_patches * PATCH]` for the positions of `h`: position `s` holds patches `s + 1` to
        `s + horizon_patches`, each in the asinh space of the statistics at the end of patch `s`.
        With `anchors`, the pair of `anchor_values` at the positions of `h`, the median of each
        step adds its own learned weight of the last value and one weight per position of the
        patch median, and the other knots move with it. Under `level_head`, every knot of a
        patch moves by the gate's share of the distance between the level head's level and the
        level of the patch's own median knot (`patch_level`)."""
        n, positions = h.shape[:2]
        z = F.silu(self.head1(h)) if self.head1 is not None else h
        raw = self.head2(z).reshape(n, positions, len(LEVELS), self.cfg.horizon_patches * PATCH)
        if anchors is not None:
            last, median = anchors
            w = self.anchor_head(z)
            shift = w[..., :PATCH] * last[..., None].to(w.dtype) + w[..., PATCH:] * median[..., None].to(w.dtype)
            raw = torch.cat([raw[:, :, :MEDIAN], raw[:, :, MEDIAN : MEDIAN + 1] + shift[:, :, None].to(raw.dtype),
                             raw[:, :, MEDIAN + 1 :]], dim=2)
        knots = monotone(raw)
        if self.level_head is not None:
            level, gate = self.level_head(z).unbind(dim=-1)
            move = torch.sigmoid(gate) * (level - patch_level(knots[:, :, MEDIAN]))
            knots = knots + move[:, :, None, None].to(knots.dtype)
        return knots

    def level(self, h: torch.Tensor) -> torch.Tensor:
        """The level head's level of the next patch, `[n, positions]`, of the normalised state `h`,
        in the asinh space of each position's statistics."""
        z = F.silu(self.head1(h)) if self.head1 is not None else h
        return self.level_head(z)[..., 0]

    def flow_loss(self, h: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """The flow-matching loss of the flow head over the present target points: at a time
        and a noise draw per position, the velocity at the point between the noise and the
        target against the straight path's velocity."""
        x1 = targets
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], x1.shape[1], 1, device=x1.device, dtype=x1.dtype)
        v = self.flow(self.flow.state(h[:, : x1.shape[1]]), (1.0 - t) * x0 + t * x1, t)
        err = (v - (x1 - x0)) ** 2 * target_mask
        return err.sum() / target_mask.sum().clamp_min(1.0)

    def sample(self, h: torch.Tensor, samples: int, steps: int, generator=None) -> torch.Tensor:
        """`samples` draws of the next patch after each position of the normalised state `h [n,
        positions, d_model]`, by `steps` Euler steps of the flow from standard normal noise:
        `[n, positions, samples, PATCH]`."""
        state = self.flow.state(h)[:, :, None, :]
        x = torch.randn(h.shape[0], h.shape[1], samples, PATCH, generator=generator, device=h.device, dtype=h.dtype)
        for k in range(steps):
            t = torch.full((1, 1, 1, 1), k / steps, device=h.device, dtype=h.dtype)
            x = x + self.flow(state, x, t) / steps
        return x

    def sample_quantiles(self, h: torch.Tensor, samples: int, steps: int, generator=None) -> torch.Tensor:
        """The levels' quantiles across `samples` draws of the flow head, `[n, positions, 15,
        PATCH]`, in the layout of the quantile head."""
        draws = self.sample(h, samples, steps, generator)
        levels = torch.tensor(LEVELS, device=h.device, dtype=draws.dtype)
        return torch.quantile(draws, levels, dim=2).permute(1, 2, 0, 3)

    def embed(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None,
              band: torch.Tensor | None, positions: torch.Tensor, copy: torch.Tensor | None = None,
              sibling: torch.Tensor | None = None, history: torch.Tensor | None = None,
              state: torch.Tensor | None = None):
        """The backbone's input `[n, s, d_model]` for `inputs`, `mask` `[n, s, PATCH]`, `band`,
        `copy` and `sibling` `[n, s, 2 * PATCH]` at `positions` (`[1, s]`, or `[n, s]` per tile),
        with the channel ids and resolution rows the router reads. Under `refine_rounds`, `state
        [n, s, d_model]` is each tile's series embedding at each position, the start embedding
        when absent."""
        cfg = self.cfg
        n, s = inputs.shape[:2]
        parts = [inputs, mask]
        if cfg.band_input:
            parts.append(band if band is not None else torch.zeros(n, s, 2 * PATCH, device=inputs.device, dtype=inputs.dtype))
        if cfg.copy_input:
            parts.append(copy if copy is not None else torch.zeros(n, s, 2 * PATCH, device=inputs.device, dtype=inputs.dtype))
        if self.side:
            parts.append(sibling if sibling is not None else torch.zeros(n, s, self.side, device=inputs.device, dtype=inputs.dtype))
        x = self.embed1(torch.cat(parts, dim=2))
        if self.embed2 is not None:
            x = self.embed2(F.silu(x))
        if self.history_in is not None and history is not None:
            if cfg.history_clip:
                history = history.clamp(-cfg.history_clip, cfg.history_clip)
            x = x + self.history_in(history.to(x.dtype))
        if self.pos is not None:
            x = x + self.pos(positions)
        if self.series_start is not None:
            x = x + (self.series_start[None, None] if state is None else state).to(x.dtype)
        channel_ids = tags % TAG_BASE if tags is not None else None
        if self.tags is not None and channel_ids is not None:
            x = x + self.tags(channel_ids)[:, None]
        resolution_rows = None
        if tags is not None and cfg.resolutions:
            upper = tags // TAG_BASE
            resolution_rows = torch.where(upper == 0, cfg.resolutions - 1, upper - 1)
        if self.resolution_tags is not None and resolution_rows is not None:
            x = x + self.resolution_tags(resolution_rows)[:, None]
        return x, channel_ids, resolution_rows

    def hidden(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None = None,
               families: torch.Tensor | None = None, band: torch.Tensor | None = None,
               copy: torch.Tensor | None = None, sibling: torch.Tensor | None = None,
               group: torch.Tensor | None = None, history: torch.Tensor | None = None,
               state: torch.Tensor | None = None):
        """The backbone's normalised state `[n, SEQ, d_model]` for the inputs of `forward_full`,
        the mean balance loss and the experts chosen per MoE layer."""
        cfg = self.cfg
        n = inputs.shape[0]
        x, channel_ids, resolution_rows = self.embed(inputs, mask, tags, band, self.positions[None], copy, sibling, history, state)
        if self.recurrent is not None:
            x = x + self.recurrent_out(self.recurrent(x)[0])
        router_bias = None
        if self.router_tags is not None and channel_ids is not None:
            router_bias = self.router_tags(channel_ids)
        if self.router_resolution is not None and resolution_rows is not None:
            bias = self.router_resolution(resolution_rows)
            router_bias = bias if router_bias is None else router_bias + bias
        if router_bias is not None:
            router_bias = router_bias[:, None].expand(n, SEQ, cfg.experts).reshape(n * SEQ, cfg.experts)
        hard = None
        if cfg.hard_routing:
            if families is None:
                families = torch.full((n,), len(FAMILIES) - 1, dtype=torch.int64, device=inputs.device)
            hard = families[:, None].expand(n, SEQ).reshape(n * SEQ)
        balances, chosen = [], []
        reads = self.reads(group, mask)
        held = (mask.sum(dim=2, keepdim=True) > 0).to(x.dtype) if cfg.canon_held else None
        for layer in self.stack():
            x, routed = layer(x, self.causal, cfg.top_k, router_bias, hard, self.positions, None, reads, held)
            if routed is not None:
                balances.append(routed[0])
                chosen.append(routed[1])
        balance = torch.stack(balances).mean() if balances else torch.zeros((), device=inputs.device, dtype=x.dtype)
        return self.norm(x), balance, chosen

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None = None,
                band: torch.Tensor | None = None, copy: torch.Tensor | None = None,
                sibling: torch.Tensor | None = None, group: torch.Tensor | None = None,
                history: torch.Tensor | None = None, state: torch.Tensor | None = None) -> torch.Tensor:
        """The quantiles alone; a hard-routed student sends every tile to the `other` family."""
        return self.forward_full(inputs, mask, tags, None, band, copy, sibling, group, history, state)[0]

    def update(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None, forecast: torch.Tensor,
               end: torch.Tensor, state: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """The series embedding `[n, d_model]` after a round: the model reads the tile under
        the per-position embeddings `state [n, SEQ, d_model]` with a round's forecast beside the
        truth (`forecast [n, SEQ, 2 * PATCH]`: the medians forecast and which steps hold one) and
        adds to `last [n, d_model]`, the embedding before the round, what a linear layer makes of
        its state at `end [n]`, the last position the round forecast."""
        h = self.hidden(inputs, mask, tags, None, None, None, forecast, None, None, state)[0]
        return last + self.series_out(h[torch.arange(inputs.shape[0], device=inputs.device), end])

    def cached_pass(self, inputs: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None = None,
                    band: torch.Tensor | None = None, room: int = 0, sibling: torch.Tensor | None = None,
                    group: torch.Tensor | None = None,
                    history: torch.Tensor | None = None,
                    state: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """A dense student's `forward_full` quantiles `[n, s, 15, PATCH]` over the `s` leading
        positions of `inputs`, and the cache `next_position` reads and writes: what every layer
        of `stack` keeps (`Layer.forward`), `[n, s + room, ...]` each, after the recurrent
        layer's states when the model has one; `room` holds zeros for positions after the `s`."""
        if self.cfg.experts:
            raise ValueError("the cached pass needs a dense student")
        s = inputs.shape[1]
        x, _, _ = self.embed(inputs, mask, tags, band, self.positions[None, :s], None, sibling, history,
                             None if state is None else state[:, :s])
        states = []
        if self.recurrent is not None:
            states.append(self.recurrent(x)[0])
            x = x + self.recurrent_out(states[0])
        reads = self.reads(group, mask)
        held = (mask.sum(dim=2, keepdim=True) > 0).to(x.dtype) if self.cfg.canon_held else None
        for layer in self.stack():
            x, _ = layer(x, self.causal[:s, :s], self.cfg.top_k, None, None, self.positions[:s], states, reads, held)
        if room:
            states = [torch.cat([h, h.new_zeros(h.shape[0], room, h.shape[2])], dim=1) for h in states]
        return self.head(self.norm(x), anchor_values(inputs[:, :s], mask[:, :s]) if self.cfg.anchors else None), states

    def next_position(self, inputs: torch.Tensor, mask: torch.Tensor, band: torch.Tensor | None,
                      tags: torch.Tensor | None, p: torch.Tensor, states: list[torch.Tensor],
                      group: torch.Tensor | None = None, history: torch.Tensor | None = None,
                      rows_fn=None, state: torch.Tensor | None = None) -> torch.Tensor:
        """The quantiles `[n, 15, PATCH]` of the patch after position `p` (`[n]`, one per tile)
        when only the patch at `p` changed since `states` was filled: reads the cached
        positions before `p` and writes what each layer keeps at `p` into `states`. Under
        `across`, the tiles of a `group` stand at the same `p`. `rows_fn` computes in place of
        `next_rows`, a compiled `next_rows` for instance."""
        q, kept_rows = (rows_fn or self.next_rows)(inputs, mask, band, tags, p, states, group, history, state)
        # the cache is written here, outside `rows_fn`: compiled, a write into an input copies the whole input
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        for kept, row in zip(states, kept_rows):
            kept[rows, p] = row
        return q

    def next_rows(self, inputs: torch.Tensor, mask: torch.Tensor, band: torch.Tensor | None,
                  tags: torch.Tensor | None, p: torch.Tensor, states: list[torch.Tensor],
                  group: torch.Tensor | None = None,
                  history: torch.Tensor | None = None,
                  state: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`next_position`'s quantiles and, for each tensor of `states`, the row `[n, width]` it
        keeps at `p`; `states` is read only."""
        if self.cfg.experts:
            raise ValueError("the cached pass needs a dense student")
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        at = p[:, None]
        x, _, _ = self.embed(inputs[rows, p][:, None], mask[rows, p][:, None], tags,
                             None if band is None else band[rows, p][:, None], at, None, None,
                             None if history is None else history[rows, p][:, None],
                             None if state is None else state[rows, p][:, None])
        d = x.shape[2]
        kept_rows = []
        if self.recurrent is not None:
            memory, states = states[0], states[1:]
            before = memory.gather(1, (p - 1).clamp_min(0)[:, None, None].expand(-1, 1, memory.shape[2]))[:, 0]
            out, _ = self.recurrent(x, (before * (p > 0)[:, None])[None].contiguous())
            kept_rows.append(out[:, 0])
            x = x + self.recurrent_out(out)
        reads = self.reads(group, mask, p)
        held = (mask[rows, p].sum(dim=1, keepdim=True) > 0).to(x.dtype) if self.cfg.canon_held else 1.0
        for layer, kept in zip(self.stack(), states):
            if layer.canon_a is not None:
                a = x[:, 0] * held
                x = x + mix_back_at(kept[:, :, 2 * d : 3 * d], p, a, layer.canon_a)
            h = layer.norm1(x)
            kv = layer.keys_values(h, at)
            if layer.across:
                x = x + layer.attend_across(h, reads)
            else:
                x = x + layer.attend_cached(h, kept[:, :, : 2 * d], kv, p)
            if layer.canon_c is not None:
                c = x[:, 0] * held
                x = x + mix_back_at(kept[:, :, 3 * d :], p, c, layer.canon_c)
                kept_rows.append(torch.cat([kv[:, 0], a, c], dim=1))
            else:
                kept_rows.append(kv[:, 0])
            h = layer.norm2(x)
            x = x + layer.w_out(F.silu(layer.w_in(h)) * layer.v_in(h))
        anchors = None
        if self.cfg.anchors:
            last, median = anchor_values(inputs, mask)
            anchors = (last[rows, p][:, None], median[rows, p][:, None])
        return self.head(self.norm(x), anchors)[:, 0], kept_rows

    def stack(self) -> list[Layer]:
        """The layers in the order a pass runs them, each `loops` times."""
        return [layer for _ in range(self.cfg.loops) for layer in self.layers]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def describe(cfg: StudentConfig, teacher: str, channels: dict[str, int], widths: list[int],
             groups: list[list[str]] | None = None) -> dict:
    """The `model.json` of a student of `cfg`: `channels` maps a name to its row, `widths` lists
    the seconds per bin of each resolution row; `groups` lists the channels sharing each row when
    the rows are groups."""
    out = {"teacher": teacher, "context": CONTEXT, "patch": PATCH, "levels": list(LEVELS),
           "config": cfg.to_json(), "channels": channels, "widths": widths, "weights": "student.pt"}
    if groups is not None:
        out["groups"] = groups
    return out


def save(model: Student, description: dict, directory: str, stem: str = "student") -> None:
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, f"{stem}.pt"))
    with open(os.path.join(directory, "model.json"), "w") as f:
        json.dump(description, f, indent=2)
        f.write("\n")


def load(directory: str, device: str | torch.device = "cpu") -> tuple[dict, Student]:
    """The description and the student saved under `directory`."""
    with open(os.path.join(directory, "model.json")) as f:
        description = json.load(f)
    trained = description.get("patch", 32)
    if trained != PATCH:
        raise ValueError(f"{directory} has patches of {trained} steps and this process runs at {PATCH}: set AXIMOE_PATCH={trained}")
    model = Student(StudentConfig.from_json(description["config"]))
    weights = os.path.join(directory, description.get("weights", "student.pt"))
    if weights.endswith(".safetensors"):
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return description, model.to(device)


def rerun_at_width(directory: str) -> None:
    """Replaces this process with one at the patch width of the model saved under `directory`,
    on the same command line, when the process runs at another width."""
    try:
        with open(os.path.join(directory, "model.json")) as f:
            trained = json.load(f).get("patch", 32)
    except OSError:
        return
    if trained != PATCH:
        os.execve(sys.executable, sys.orig_argv, {**os.environ, "AXIMOE_PATCH": str(trained)})


def widen(model: Student, cfg: StudentConfig) -> Student:
    """A student of `cfg` with `model`'s weights and zero weights from the inputs `cfg` adds
    after `model`'s in the patch embedding, so it computes what `model` does until trained."""
    wide = Student(cfg)
    state = model.state_dict()
    narrow = state["embed1.weight"]
    weight = torch.zeros_like(wide.embed1.weight)
    weight[:, : narrow.shape[1]] = narrow
    wide.load_state_dict(wide.state_dict() | state | {"embed1.weight": weight})
    return wide


def without_flow(description: dict, model: Student) -> tuple[dict, Student]:
    """The description and the student of `model` without its flow head; the backbone and the
    quantile head keep their weights."""
    plain = Student(dataclasses.replace(model.cfg, flow_head=False))
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("flow.")})
    return description | {"config": plain.cfg.to_json(), "weights": "student.pt"}, plain


def tag_id(description: dict, channel: str | None, seconds: int | None) -> int:
    """The packed tag id of a series of channel `channel`, if known, at `seconds` per bin, if
    known; any other name or width uses the `unknown` row."""
    cfg = description["config"]
    unknown_channel = max(cfg["tags"] - 1, 0)
    row = description["channels"].get(channel, unknown_channel) if channel is not None else unknown_channel
    widths = description["widths"]
    res = widths.index(seconds) if seconds in widths else len(widths)
    return row + (res + 1) * TAG_BASE


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a model directory without the model's flow head.")
    ap.add_argument("model", help="model directory with a flow head")
    ap.add_argument("out", help="directory of the model without it")
    args = ap.parse_args(argv)
    description, model = load(args.model)
    if model.flow is None:
        raise SystemExit(f"{args.model} has no flow head")
    description, plain = without_flow(description, model)
    save(plain, description, args.out)
    print(f"{args.out}: {args.model} without its flow head, {plain.num_params():,} of {model.num_params():,} parameters")


if __name__ == "__main__":
    main()
