"""The unroll's one-position passes in fewer kernels, with the numbers of `Student.next_rows`.

A pass computes one position of 64 tiles, so its time on a card is its count of kernels. Three things are taken
out of every pass:

* The weight of each matrix product is cast to the passes' precision once per step (`Bundle.refresh`), where
  autocast casts it in every pass.
* The two input maps of a layer's feed-forward block are one product.
* Keys and values are read from a cache of their own, in BF16 when the passes run in BF16 and the model has no
  rotary positions: the cached numbers are BF16 numbers then, and a pass reads half the bytes.

Against `Student.next_rows` compiled, a pass rounds to half precision in other places, so a share of the patches
fed back differs by one half-precision step: on an A40 0.02 to 0.07 % in BF16 and 0.35 % in FP16, on an RTX 2080 Ti
7.7 % in FP16. Both passes are equally far from the same unroll in single precision (`train.py --pass-check`).
Two switches: AXIMOE_FAST_HALF=1 keeps the cache in the passes' precision in FP16 too (A40: 5.8 % of the patches
differ, RTX 2080 Ti: 2.0 %), AXIMOE_FAST_MERGE=1 also computes the query, key and value maps as one product (A40,
BF16: 10 % differ); AXIMOE_FAST_HALF=0 and AXIMOE_FAST_MERGE=0 switch each change off.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from aximoe.student import LEVELS, PATCH, POSITIONS, TAG_BASE, Student, mix_back_at, monotone, rotate


MERGE_QKV = os.environ.get("AXIMOE_FAST_MERGE", "w") in ("1", "q")
MERGE_WV = os.environ.get("AXIMOE_FAST_MERGE", "w") in ("1", "w")
HALF = os.environ.get("AXIMOE_FAST_HALF", "auto")


def supports(model: Student) -> bool:
    """Whether `Bundle` covers every part of the model's one-position pass."""
    cfg = model.cfg
    return bool(cfg.band_input and cfg.horizon_patches == 1 and cfg.loops == 1 and model.side == 0
                and model.embed2 is not None and model.head1 is not None and model.history_in is None
                and model.series_start is None
                and not (cfg.experts or cfg.recurrent or cfg.across or cfg.anchors or cfg.level_head or cfg.flow_head
                         or cfg.copy_input or cfg.qk_norm or cfg.diff_attention or cfg.canon_held))


def fixed(shape, dtype: torch.dtype, device) -> torch.Tensor:
    """A tensor at an address that a CUDA graph reads and writes in place."""
    t = torch.empty(shape, dtype=dtype, device=device)
    torch._dynamo.mark_static_address(t)
    return t


class Bundle:
    """What the one-position passes of one step read beside the model: each product's weight in the passes'
    precision `dtype` and the caches the first pass fills."""

    def __init__(self, model: Student, dtype: torch.dtype):
        self.model, self.dtype = model, dtype
        self.canon = model.layers[0].canon_a is not None
        # rotated keys are single-precision numbers, and FP16 passes store sums that are no FP16 numbers
        half = HALF == "1" or (HALF == "auto" and dtype == torch.bfloat16)
        self.kv_dtype = dtype if half and not model.cfg.rope else torch.float32
        device = model.head2.weight.device
        like = lambda w: fixed(w.shape, dtype, device)  # noqa: E731
        self.embed1, self.embed2 = like(model.embed1.weight), like(model.embed2.weight)
        self.head1, self.head2 = like(model.head1.weight), like(model.head2.weight)
        self.in_proj, self.out_proj, self.wv, self.w_out = [], [], [], []
        for layer in model.layers:
            ff, d = layer.w_in.weight.shape
            self.in_proj.append(like(layer.attn.in_proj_weight))
            self.out_proj.append(like(layer.attn.out_proj.weight))
            self.wv.append(fixed((2 * ff, d), dtype, device))
            self.w_out.append(like(layer.w_out.weight))
        self.caches: dict = {}

    @torch.no_grad()
    def refresh(self) -> None:
        """Copies the model's present weights into the bundle."""
        m = self.model
        self.embed1.copy_(m.embed1.weight)
        self.embed2.copy_(m.embed2.weight)
        self.head1.copy_(m.head1.weight)
        self.head2.copy_(m.head2.weight)
        for i, layer in enumerate(m.layers):
            ff = layer.w_in.weight.shape[0]
            self.in_proj[i].copy_(layer.attn.in_proj_weight)
            self.out_proj[i].copy_(layer.attn.out_proj.weight)
            self.wv[i][:ff].copy_(layer.w_in.weight)
            self.wv[i][ff:].copy_(layer.v_in.weight)
            self.w_out[i].copy_(layer.w_out.weight)

    def cache(self, cached_pass):
        """`cached_pass` (`Student.cached_pass`) after a `refresh`, its cache split into tensors at fixed
        addresses: per layer the keys and values, then, under `canon`, what the two mixes read."""

        def run(*args, **kwargs):
            self.refresh()
            full, states = cached_pass(*args, **kwargs)
            d = self.model.cfg.d_model
            key = tuple((s.shape, s.dtype) for s in states)
            if key not in self.caches:
                held = []
                for s in states:
                    held.append(fixed((*s.shape[:2], 2 * d), self.kv_dtype, s.device))
                    if self.canon:
                        held.append(fixed((*s.shape[:2], 2 * d), s.dtype, s.device))
                self.caches[key] = held
            held = self.caches[key]
            for i, s in enumerate(states):
                if self.canon:
                    held[2 * i].copy_(s[..., : 2 * d])
                    held[2 * i + 1].copy_(s[..., 2 * d :])
                else:
                    held[i].copy_(s)
            return full, held

        return run

    def step(self, fed_back):
        """Pass `t` of an unroll as `train.unroll_step` runs it, through `rows`; `states` are the caches of
        `cache`, `fed_back` is `train.fed_back`."""

        def run(inputs, mask, band, tags, t, cut, j, states, group, history, state, loc, scale, picks, limit):
            p = (cut + t - 1).clamp(max=POSITIONS - 2)
            q, kept_rows = self.rows(inputs, mask, band, tags, p, states)
            rows = torch.arange(inputs.shape[0], device=inputs.device)
            median, around = fed_back(q[:, picks].float(), loc[rows, p], scale[rows, p], loc[rows, p + 1], scale[rows, p + 1], limit)
            keep, s = (j < t)[:, None], p + 1
            inputs[rows, s] = torch.where(keep, inputs[rows, s], median.to(inputs.dtype))
            mask[rows, s] = torch.where(keep, mask[rows, s], 1.0)
            band[rows, s] = torch.where(keep, band[rows, s], around.to(band.dtype))
            for cache, row in zip(states, kept_rows):
                cache[rows, p] = row

        return run

    def rows(self, inputs: torch.Tensor, mask: torch.Tensor, band: torch.Tensor, tags: torch.Tensor | None,
             p: torch.Tensor, states: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The quantiles `[n, 15, PATCH]` of the patch after position `p [n]`, as `Student.next_rows` computes
        them, and for each cache of `states` the row it keeps at `p`."""
        m = self.model
        cfg = m.cfg
        n, d, heads = inputs.shape[0], cfg.d_model, cfg.heads
        rows = torch.arange(n, device=inputs.device)
        at = p[:, None]
        x = F.linear(torch.cat([inputs[rows, p][:, None], mask[rows, p][:, None], band[rows, p][:, None]], dim=2),
                     self.embed1, m.embed1.bias)
        x = F.linear(F.silu(x), self.embed2, m.embed2.bias)
        if m.pos is not None:
            x = x + m.pos(at)
        if m.tags is not None and tags is not None:
            x = x + m.tags(tags % TAG_BASE)[:, None]
        if m.resolution_tags is not None and tags is not None:
            upper = tags // TAG_BASE
            x = x + m.resolution_tags(torch.where(upper == 0, cfg.resolutions - 1, upper - 1))[:, None]
        kept_rows = []
        for i, layer in enumerate(m.layers):
            kept = states[2 * i] if self.canon else states[i]
            if self.canon:
                mixed = states[2 * i + 1]
                a = x[:, 0]
                x = x + mix_back_at(mixed[:, :, :d], p, a, layer.canon_a)
            if MERGE_QKV:
                # one product for the three maps; each part then adds its own bias, as its own product would
                qkv = F.linear(layer.norm1(x), self.in_proj[i])
                bias = layer.attn.in_proj_bias
                q = (qkv[..., :d] + bias[:d].to(qkv.dtype)).view(n, 1, heads, -1).transpose(1, 2)
                kv = qkv[..., d:] + bias[d:].to(qkv.dtype)
            else:
                h = layer.norm1(x)
                kv = F.linear(h, self.in_proj[i][d:], layer.attn.in_proj_bias[d:])
                q = F.linear(h, self.in_proj[i][:d], layer.attn.in_proj_bias[:d]).view(n, 1, heads, -1).transpose(1, 2)
            if layer.turn is not None:
                k = rotate(kv[..., :d].view(n, 1, heads, -1).transpose(1, 2), at, layer.turn)
                kv = torch.cat([k.transpose(1, 2).reshape(n, 1, d), kv[..., d:]], dim=2)
                q = rotate(q, at, layer.turn)
            q = q[:, :, 0]
            S = kept.shape[1]
            keys, values = kept[..., :d].view(n, S, heads, -1), kept[..., d:].view(n, S, heads, -1)
            own_k, own_v = kv[:, 0, :d].view(n, heads, -1), kv[:, 0, d:].view(n, heads, -1)
            later = (torch.arange(S, device=p.device)[None] >= p[:, None])[:, :, None]
            scale = q.shape[-1] ** -0.5
            # a softmax over the cached positions and the own one, kept apart so the cache is read in place
            cached = ((keys.float() * q[:, None].float()).sum(-1) * scale).masked_fill(later, float("-inf"))
            own = (own_k * q).sum(-1) * scale
            top = torch.maximum(cached.amax(1), own)
            e_cached, e_own = torch.exp(cached - top[:, None]), torch.exp(own - top)
            total = e_cached.sum(1) + e_own
            out = (e_cached[..., None] * values.float()).sum(1) / total[..., None] + (e_own / total)[..., None] * own_v
            x = x + F.linear(out.reshape(n, 1, d), self.out_proj[i], layer.attn.out_proj.bias)
            kept_rows.append(kv[:, 0])
            if self.canon:
                c = x[:, 0]
                x = x + mix_back_at(mixed[:, :, d:], p, c, layer.canon_c)
                kept_rows.append(torch.cat([a, c], dim=1))
            if MERGE_WV:
                wv = F.linear(layer.norm2(x), self.wv[i])
                ff = wv.shape[-1] // 2
                gate = F.silu(wv[..., :ff] + layer.w_in.bias.to(wv.dtype)) * (wv[..., ff:] + layer.v_in.bias.to(wv.dtype))
                x = x + F.linear(gate, self.w_out[i], layer.w_out.bias)
            else:
                h, ff = layer.norm2(x), self.wv[i].shape[0] // 2
                gate = F.silu(F.linear(h, self.wv[i][:ff], layer.w_in.bias)) * F.linear(h, self.wv[i][ff:], layer.v_in.bias)
                x = x + F.linear(gate, self.w_out[i], layer.w_out.bias)
        z = F.silu(F.linear(m.norm(x), self.head1, m.head1.bias))
        raw = F.linear(z, self.head2, m.head2.bias).reshape(n, 1, len(LEVELS), PATCH)
        return monotone(raw)[:, 0], kept_rows
