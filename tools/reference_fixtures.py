"""Generate parity fixtures by executing a pinned, unmodified NX-AI checkout.

Usage: python tools/reference_fixtures.py /path/to/xlstm
Requires torch, numpy, einops, packaging. No Python is needed to run Cargo tests.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

REVISION = "ab22eadbd245f293dd8dec38ed29963d73758a12"
reference = Path(sys.argv[1]).resolve()
actual = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
if actual != REVISION:
    raise SystemExit(f"Expected reference {REVISION}, got {actual}")
dirty = subprocess.check_output(
    ["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=normal", "--", "xlstm"],
    text=True,
).strip()
if dirty:
    raise SystemExit("Reference xlstm sources must be unmodified, with no untracked source files")
sys.path.insert(0, str(reference))

from xlstm.blocks.mlstm.backends import parallel_stabilized_simple, recurrent_step_stabilized_simple
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.blocks.slstm.src.vanilla.slstm import slstm_forward_pointwise
from xlstm.blocks.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from xlstm.components.feedforward import FeedForwardConfig
from xlstm.components.ln import LayerNorm

torch.set_num_threads(1)
torch.manual_seed(271828)


def pack(tensor):
    tensor = tensor.detach().cpu().float().contiguous()
    return {"shape": list(tensor.shape), "values": tensor.flatten().tolist()}


def export(module):
    return {name: pack(param) for name, param in module.named_parameters()}


def gradients(module):
    return {name: pack(param.grad) for name, param in module.named_parameters() if param.grad is not None}


def randomize(module):
    # Nonzero recurrent/gate weights are essential: initialization alone cannot
    # expose missing recurrence or incorrect gate inputs in a port.
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.normal_(0.0, 0.17)
            if "learnable_skip" in name:
                param.add_(1.0)


def make_block(kind, index, *, width=8, heads=2, time=5, kernel=3, group=4, depth=3):
    if kind == "m":
        return xLSTMBlock(xLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(embedding_dim=width, num_heads=heads, context_length=time,
                conv1d_kernel_size=kernel, qkv_proj_blocksize=group, round_proj_up_to_multiple_of=1),
            _num_blocks=depth, _block_idx=index))
    return xLSTMBlock(xLSTMBlockConfig(
        slstm=sLSTMLayerConfig(embedding_dim=width, num_heads=heads, backend="vanilla", dtype="float32",
            conv1d_kernel_size=kernel),
        feedforward=FeedForwardConfig(embedding_dim=width, proj_factor=1.5, round_proj_up_to_multiple_of=1),
        _num_blocks=depth, _block_idx=index))


blocks = [make_block(kind, index) for index, kind in enumerate("msm")]
stack = torch.nn.Sequential(*blocks, LayerNorm(ndim=8))
randomize(stack)
x = torch.randn(2, 5, 8, requires_grad=True)
loss_weights = torch.randn(2, 5, 8)
y = stack(x)
(y * loss_weights).sum().backward()
fixtures = {
    "reference_commit": REVISION,
    "torch_version": torch.__version__,
    "stack": {"input": pack(x), "weights": export(stack), "output": pack(y),
              "loss_weights": pack(loss_weights), "input_gradient": pack(x.grad),
              "parameter_gradients": gradients(stack)},
}

# Test both matrix kernels against the actual reference functions, including
# q/k/v and gate derivatives for an asymmetric multi-head, multi-batch case.
q, k, v = [torch.randn(2, 2, 5, 3, requires_grad=True) for _ in range(3)]
i, f = [torch.randn(2, 2, 5, 1, requires_grad=True) for _ in range(2)]
weights = torch.randn(2, 2, 5, 3)
parallel = parallel_stabilized_simple(q, k, v, i, f)
(parallel * weights).sum().backward()
c, n, m = torch.zeros(2, 2, 3, 3), torch.zeros(2, 2, 3, 1), torch.zeros(2, 2, 1, 1)
recurrent_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v, i, f)]
outputs = []
for t in range(5):
    output, (c, n, m) = recurrent_step_stabilized_simple(c, n, m,
        *(x[:, :, t:t+1].clone() for x in recurrent_inputs))
    outputs.append(output)
recurrent = torch.cat(outputs, dim=2)
(recurrent * weights).sum().backward()
fixtures["mlstm"] = {
    "query": pack(q), "key": pack(k), "value": pack(v), "input": pack(i), "forget": pack(f),
    "parallel": pack(parallel), "recurrent": pack(recurrent),
    "memory": pack(c), "normalizer": pack(n), "log_scale": pack(m), "loss_weights": pack(weights),
    "gradients": {name: pack(tensor.grad) for name, tensor in zip(["query", "key", "value", "input", "forget"], [q, k, v, i, f])},
    "recurrent_gradients": {name: pack(tensor.grad) for name, tensor in zip(["query", "key", "value", "input", "forget"], recurrent_inputs)},
}

# Pointwise scalar recurrence: raw gates and all four states are checked.
gates = torch.randn(5, 2, 4, 6, requires_grad=True)
state = torch.zeros(4, 2, 6)
states = []
for t in range(5):
    state, _ = slstm_forward_pointwise(gates[t].reshape(2, 24), torch.zeros(2, 24),
                                     torch.zeros(24), state, {})
    states.append(state)
output = torch.stack([s[0] for s in states], dim=1)
weights = torch.randn(2, 5, 6)
(output * weights).sum().backward()
fixtures["slstm"] = {"gates": pack(gates), "output": pack(output), "state": pack(state),
                     "loss_weights": pack(weights), "gate_gradients": pack(gates.grad)}

destination = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nxai.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(fixtures, indent=2) + "\n")
print(f"Wrote {destination} ({destination.stat().st_size} bytes)")

# Stratified shapes with independent seeds and fully randomized learned weights.
# The pinned reference's convolution has bugs for kernels 0 (mLSTM) and 1;
# those supported Burn edge cases are checked separately with streaming tests.
cases = [
    (101, "m",   1, 1, 4, 1, 2, 1),
    (211, "s",   2, 2, 6, 2, 0, 2),
    (307, "ms",  3, 3, 8, 4, 7, 4),
    (401, "sm",  1, 7, 12, 3, 3, 6),
    (503, "mm",  2, 11, 6, 3, 2, 3),
    (601, "ss",  3, 17, 8, 2, 4, 2),
    (709, "msm", 1, 31, 12, 2, 3, 4),
    (809, "sms", 2, 7, 16, 4, 4, 2),
    (907, "m",   3, 17, 8, 1, 7, 8),
    (1009,"s",   1, 31, 12, 3, 0, 3),
    (1103,"ms",  2, 11, 4, 2, 2, 2),
    (1201,"sm",  2, 3, 16, 2, 3, 4),
]
sweep = []
for seed, kinds, batch, time, width, heads, kernel, group in cases:
    torch.manual_seed(seed)
    blocks = [make_block(kind, index, width=width, heads=heads, time=time,
                        kernel=kernel, group=group, depth=len(kinds))
              for index, kind in enumerate(kinds)]
    model = torch.nn.Sequential(*blocks, LayerNorm(ndim=width))
    randomize(model)
    x = torch.randn(batch, time, width, requires_grad=True)
    loss_weights = torch.randn_like(x)
    y = model(x)
    (y * loss_weights).sum().backward()
    sweep.append({
        "seed": seed, "kinds": kinds, "width": width, "heads": heads,
        "kernel": kernel, "group": group,
        "input": pack(x), "output": pack(y), "loss_weights": pack(loss_weights),
        "input_gradient": pack(x.grad), "weights": export(model),
        "parameter_gradients": gradients(model),
    })
destination = destination.with_name("nxai_sweep.json")
destination.write_text(json.dumps({"reference_commit": REVISION,
    "torch_version": torch.__version__, "cases": sweep}, separators=(",", ":")) + "\n")
print(f"Wrote {destination} ({destination.stat().st_size} bytes; {len(sweep)} cases)")
