"""Muon for the student's hidden matrices, after "Muon is Scalable for LLM Training" (arXiv 2502.16982).

For a weight matrix W with gradient G, learning rate lr, weight decay wd, momentum mu and scale s:
M = mu M + G (M starts at zero), U = G + mu M; X0 = U / (|U|_F + 1e-7), then five Newton-Schulz steps
X = a X + b (X X^T) X + c (X X^T)^2 X, a tall matrix transposed so that X X^T is the small Gram matrix;
W = (1 - lr wd) W - lr s sqrt(max(rows, columns)) X. Everything in float32.

A parameter tensor can hold several matrices stacked by rows (the attention's input map holds query, key
and value): its group's `blocks` says how many, and each block is orthogonalised and scaled on its own.
The Newton-Schulz steps run once per step for all blocks of one shape, as one batch of matrix products.
"""
from __future__ import annotations

import math
import re

import torch

A, B, C = 3.4445, -4.7750, 2.0315
NS_STEPS = 5
# the hidden matrices of every layer: the attention's input and output maps and the three feed-forward maps
HIDDEN = re.compile(r"layers\.\d+\.(attn\.in_proj_weight|attn\.out_proj\.weight|w_in\.weight|v_in\.weight|w_out\.weight)")
# the attention's input map stacks the query, key and value maps by rows
IN_PROJ = "attn.in_proj_weight"


def newton_schulz(x: torch.Tensor) -> torch.Tensor:
    """Five Newton-Schulz steps on each matrix of the stack `x` (n, rows, columns), rows <= columns."""
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=B, alpha=C), x, beta=A)
    return x


def orthogonalise(blocks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Newton-Schulz on every matrix of `blocks`, each a stack (k, rows, columns), batched over the blocks
    of one shape; a tall shape is transposed for the iteration and back."""
    shapes: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(blocks):
        shapes.setdefault(tuple(sorted(b.shape[-2:])), []).append(i)
    out: list[torch.Tensor] = [None] * len(blocks)
    for members in shapes.values():
        tall = [blocks[i].shape[-2] > blocks[i].shape[-1] for i in members]
        x = newton_schulz(torch.cat([blocks[i].mT if t else blocks[i] for i, t in zip(members, tall)]))
        start = 0
        for i, t in zip(members, tall):
            o = x[start:start + blocks[i].shape[0]]
            out[i] = o.mT if t else o
            start += blocks[i].shape[0]
    return out


def split(model: torch.nn.Module) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Muon's parameter groups (the attention's input maps as three blocks each, the other hidden matrices
    as one) and every other parameter, for AdamW."""
    stacked, single, rest = [], [], []
    for name, p in model.named_parameters():
        m = HIDDEN.fullmatch(name)
        (rest if m is None else stacked if m.group(1) == IN_PROJ else single).append(p)
    return [{"params": stacked, "blocks": 3}, {"params": single, "blocks": 1}], rest


def describe(groups: list[dict], rest: list[torch.nn.Parameter], momentum: float, scale: float) -> str:
    tensors = [p for g in groups for p in g["params"]]
    blocks = sum(len(g["params"]) * g["blocks"] for g in groups)
    return (f" on {len(tensors)} hidden matrices ({blocks} blocks, {sum(p.numel() for p in tensors)} parameters), "
            f"momentum {momentum}, scale {scale}; AdamW on the other {len(rest)} tensors ({sum(p.numel() for p in rest)} parameters)")


def skip_together(scaler: torch.amp.GradScaler, optimizers: list[torch.optim.Optimizer]) -> None:
    """After `scaler.unscale_` of each optimizer: a step with a non-finite gradient in any of them is skipped
    by all, as by one optimizer over every parameter, since the clip over all parameters carries one
    optimizer's non-finite norm into every other's gradients."""
    if not scaler.is_enabled():
        return
    states = [scaler._per_optimizer_states[id(o)]["found_inf_per_device"] for o in optimizers]
    # the fused AdamW step reads the flag as 0 or 1
    found = (sum(v for s in states for v in s.values()) > 0).to(torch.float32)
    for s in states:
        for device in s:
            s[device] = found


class Muon(torch.optim.Optimizer):
    """Muon over 2-D parameters; a group's `blocks` splits each of its tensors into that many matrices by
    rows. Under a GradScaler the skip of a step with non-finite gradients is decided on the device."""

    # the scaler hands step() its skip flag as `found_inf`, a tensor on the device
    _step_supports_amp_scaling = True

    def __init__(self, params, lr: float, momentum: float = 0.95, scale: float = 0.2, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, scale=scale, weight_decay=weight_decay, blocks=1))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2 or p.shape[0] % group["blocks"]:
                    raise ValueError(f"Muon takes matrices whose rows split into {group['blocks']} blocks, not {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        if getattr(self, "grad_scale", None) is not None:
            raise RuntimeError("Muon reads unscaled gradients: GradScaler.unscale_ comes before GradScaler.step")
        found = getattr(self, "found_inf", None)
        skip = None if found is None else found > 0
        work = []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            for p in params:
                if not self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
            bufs = [self.state[p]["momentum_buffer"] for p in params]
            mu = group["momentum"]
            if skip is None:
                torch._foreach_mul_(bufs, mu)
                torch._foreach_add_(bufs, grads)
                nesterov = torch._foreach_add(grads, bufs, alpha=mu)
            else:
                # on a skipped step the buffers stay as they are and the orthogonalised update is zero
                grads = [torch.where(skip, 0.0, g) for g in grads]
                torch._foreach_mul_(bufs, torch.where(skip, 1.0, mu))
                torch._foreach_add_(bufs, grads)
                nesterov = torch._foreach_mul(bufs, torch.where(skip, 0.0, mu))
                torch._foreach_add_(nesterov, grads)
            k = group["blocks"]
            work.append((group, params, [u.view(k, u.shape[0] // k, u.shape[1]) for u in nesterov]))
        outs = iter(orthogonalise([b for _, _, blocks in work for b in blocks]))
        for group, params, blocks in work:
            lr, wd, s = group["lr"], group["weight_decay"], group["scale"]
            torch._foreach_mul_(params, 1.0 - lr * wd if skip is None else torch.where(skip, 1.0, 1.0 - lr * wd))
            # one update call per block shape, whose larger side sets the gain
            by_gain: dict[float, tuple[list, list]] = {}
            for p, b in zip(params, blocks):
                ps, us = by_gain.setdefault(math.sqrt(max(b.shape[-2:])), ([], []))
                ps.append(p)
                us.append(next(outs).reshape(p.shape))
            for gain, (ps, us) in by_gain.items():
                torch._foreach_add_(ps, us, alpha=-lr * s * gain)
        return None
