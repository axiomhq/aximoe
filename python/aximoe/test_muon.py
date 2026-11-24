"""Test of the step AdamW and Muon skip together under a loss scaler.

    python -I python/aximoe/test_muon.py
"""
import os
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aximoe import muon


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].w_in = torch.nn.Linear(32, 48, bias=False)
        self.layers[0].w_out = torch.nn.Linear(48, 32, bias=False)
        self.head = torch.nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.layers[0].w_out(torch.nn.functional.silu(self.layers[0].w_in(x))))


@unittest.skipUnless(torch.cuda.is_available(), "the loss scaler and the fused AdamW step need a card")
class SkipTogether(unittest.TestCase):
    def test_a_step_both_optimizers_flag_leaves_every_weight_as_it_was(self):
        torch.manual_seed(0)
        model = Net().cuda()
        groups, rest = muon.split(model)
        optimizers = [torch.optim.AdamW(rest, lr=1e-3, fused=True),
                      muon.Muon([g for g in groups if g["params"]], lr=1e-3, scale=0.4)]
        scaler = torch.amp.GradScaler("cuda")
        before = [p.detach().clone() for p in model.parameters()]
        for step in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(torch.randn(8, 32, device="cuda")).float().pow(2).mean()
            if step == 0:
                loss = loss * float("nan")
            scaler.scale(loss).backward()
            for o in optimizers:
                scaler.unscale_(o)
            muon.skip_together(scaler, optimizers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in optimizers:
                scaler.step(o)
            scaler.update()
            for o in optimizers:
                o.zero_grad(set_to_none=True)
            if step == 0:
                for p, b in zip(model.parameters(), before):
                    torch.testing.assert_close(p, b, rtol=0.0, atol=0.0)
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        self.assertTrue(any(float((p - b).abs().max()) > 0 for p, b in zip(model.parameters(), before)))


if __name__ == "__main__":
    unittest.main()
