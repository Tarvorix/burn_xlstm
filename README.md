# burn-xlstm

A Rust/Burn implementation of the original 2024 **xLSTM** architecture: stabilized
scalar-memory sLSTM, matrix-memory mLSTM, and pre-normalized residual stacks.

References:

- [Beck et al., xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm), pinned for parity tests at
  [`ab22ead`](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12)

Requires Rust 1.95+ and Burn 0.22.1. Burn 0.22 uses runtime `Device` selection;
models and tensors have no backend type parameter.

## Usage

The crate is available from this Git repository; it has not been published to crates.io.

```toml
[dependencies]
burn = { version = "0.22.1", default-features = false, features = ["std", "flex"] }
burn-xlstm = { git = "https://github.com/Tarvorix/burn_xlstm.git" }
```

```rust
use burn::tensor::{Device, Tensor};
use burn_xlstm::stack::{BlockKind, XlstmConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let device = Device::flex();
let model = XlstmConfig::new(
    32,
    vec![BlockKind::Mlstm, BlockKind::Slstm, BlockKind::Mlstm],
).with_num_heads(4).init(&device)?;

// Sequence layout is always [batch, time, features].
let input = Tensor::<3>::ones([2, 16, 32], &device);
let output = model.forward(input);
assert_eq!(output.dims(), [2, 16, 32]);

// Start a stream with None; retain every block's memory and convolution history.
let token = Tensor::<3>::ones([2, 1, 32], &device);
let (_, state) = model.step(token.clone(), None);
let (next, state) = model.step(token, Some(state));
assert_eq!(next.dims(), [2, 1, 32]);
let state = state.detach(); // optional: truncated backpropagation boundary
# let _ = state;
# Ok(())
# }
```

`forward` starts a fresh sequence and uses a causal parallel mLSTM kernel.
`forward_with_state` accepts arbitrary nonempty chunks and uses recurrent mLSTM.
`step` requires time length one. Pass `None` to reset a sequence. State must match
the model, batch, device, and dtype; it retains its backward graph until detached.

Use `Device::flex().autodiff()` for training. Burn's `Module::valid()` produces an
inference model with dropout disabled; `Module::train()` enables training again.
Sequence/stream comparisons require dropout to be disabled. Full and recurrent
mLSTM can differ slightly because their log stabilizers and epsilon scales differ.

## Implemented structure

| Component | Behavior |
| --- | --- |
| sLSTM | Exponential input gate, sigmoid forget gate, stabilized scalar memory/normalizer, sigmoid output gate, headwise recurrent memory mixing |
| sLSTM layer | Causal depthwise convolution + `SiLU` for input/forget projections; unconvolved candidate/output projections; dropout then head normalization |
| sLSTM block | Pre-LayerNorm memory residual, followed by a separate pre-LayerNorm GELU-gated feedforward residual |
| mLSTM | Stabilized key/value covariance update, scaled keys, absolute normalizer with floor; recurrent and causal parallel execution |
| mLSTM layer | Expanded memory/output branches; convolution; grouped q/k/v projections; head normalization; learned local skip; `SiLU` output gate; down-projection |
| Stack | Explicit mixed block order and optional final `LayerNorm` |
| Language model | Token embeddings, mixed stack, untied bias-free vocabulary logits |

Normalization acts independently at each token. Head normalization never mixes
different heads or timesteps. Q/k/v projection groups are independent of memory
head count. Affine normalization uses the reference's `1 + weight` parameterization.
GELU is composed from erf tensor operations so its backward pass differentiates
the same function on CPU and GPU.

Defaults follow the original reference: four memory heads, convolution width four,
q/k/v projection group width four, mLSTM expansion 2× and sLSTM feedforward expansion
1.3×, both rounded up to multiples of 64. Explicit inner widths support smaller
models. Input weights use Nguyen/Salazar initialization; output projections use
depth-dependent Wang initialization. sLSTM recurrence starts at zero with
block-dependent power-law forget biases; mLSTM forget biases span 3 to 6.

`functional` exposes the equations without learned projections; `cells` adds
learned gates/recurrence; `layers`, `stack`, and `lm` provide progressively higher
level models. Config initialization returns `ConfigError` for invalid settings.
Input/state shape contract violations panic, consistent with Burn tensors.

## Training and checkpoints

```sh
cargo run --example delayed_copy
```

The example overfits a small delayed-token-copy batch using Adam and both memory
types. It is a training smoke test, not a generalization or language-model benchmark.

Models derive Burn's `Module` and support optimizers, device movement, autodiff,
and Burn records:

```rust,no_run
use burn::{module::Module, store::ModuleRecord, tensor::Device};
use burn_xlstm::stack::{BlockKind, XlstmConfig};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let device = Device::flex();
let config = XlstmConfig::new(32, vec![BlockKind::Mlstm, BlockKind::Slstm]);
let model = config.init(&device)?;
model.into_record().save("model.bpk")?;
let restored = config.init(&device)?.load_record(ModuleRecord::load("model.bpk")?);
# let _ = restored;
# Ok(())
# }
```

Save the configuration separately with Burn's `Config` serialization. Records
contain learned parameters, not the current stream's recurrent/convolution state.

## Validation

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo doc --no-deps
cargo check --no-default-features
```

Committed fixtures are generated by executing the unmodified pinned NX-AI Python
code on CPU float32. Cargo tests require no Python installation. Tests compare:

- sLSTM outputs, all state tensors, and gate gradients;
- mLSTM parallel/recurrent outputs, final state, and q/k/v/gate gradients;
- a full mLSTM–sLSTM–mLSTM stack plus 12 randomized model configurations,
  input gradients, and every parameter gradient, through full and chunked execution;
- independent f64 equations and numerical finite-difference gradients;
- token/chunk continuation, causality, convolution edge cases, state detachment,
  record serialization, extreme logits, sequences up to 2,048 tokens, and actual
  optimizer learning on unseen sequences with convolution disabled.

The fixture stack uses nonzero recurrent and gate weights. Typical scaled
tolerances are `2e-6` for raw outputs and up to `8e-5` for full-stack gradients.
The fixture importer explicitly maps the reference sLSTM layer's swapped
`fgate`/`igate` attribute names, its packed recurrent matrices, and tensor layouts.

The full reference sweep and every parameter gradient also pass the same tolerances
on this development machine's macOS WGPU backend in explicit float32. That test
requires a GPU and is opt-in:

```sh
cargo test --features wgpu --test parity wgpu_stack -- --ignored
```

The default suite passes 24 integration tests and 2 doctests. Scalar-only,
matrix-only, and mixed models reached 100% accuracy on the held-out delayed-copy
test, including sequences twice the training length. This is evidence for the
tested configurations, not a guarantee of universal correctness. See the
[validation record](https://github.com/Tarvorix/burn_xlstm/blob/main/docs/VALIDATION.md)
for settings, tolerances, measured results, and limitations.

To regenerate fixtures, install `tools/requirements-reference.txt` in a separate
Python environment, check out the pinned revision, then run:

```sh
python tools/reference_fixtures.py /path/to/NX-AI-xlstm
```

## Backends and limits

`flex` is the default pure Rust CPU backend. Enable `wgpu` for Burn's portable GPU
backend and select `Device::wgpu(Default::default())`. To select explicit float32,
call `device.configure(burn::tensor::FloatDType::F32)` on a mutable device before
creating any tensors or model parameters. Other Burn backends can be
enabled through a direct Burn dependency. `--no-default-features` builds the
library without an execution backend.

This is a portable tensor implementation of the original architecture. It does
not include NX-AI's custom CUDA/Triton kernels, xLSTM-large/7B variants, a parallel
chunkwise mLSTM kernel, packed/ragged sequences, padding masks, weight tying, or a
general `PyTorch` checkpoint converter. Bias-free projections and GELU feedforwards
match the selected reference configuration. GPU throughput and mixed-precision
training have not been benchmarked or certified; float32 CPU is the numerical
baseline. Parallel mLSTM uses O(time²) intermediate storage; use the stateful API
for bounded recurrent memory during inference. sLSTM necessarily runs recurrently.

Two numerical edge cases are handled deliberately: empty sLSTM lanes initialize
their scale independently, and readout avoids directly evaluating `exp(-m)` when
`m` is very negative. One-dimensional scalar heads use the finite left endpoint
of the reference's forget-bias schedule.

## License

Project code: [MIT](https://github.com/Tarvorix/burn_xlstm/blob/main/LICENSE). The xLSTM port is adapted from NX-AI's
Apache-2.0-licensed implementation; its applicable notices and license are retained
in [NOTICE](https://github.com/Tarvorix/burn_xlstm/blob/main/NOTICE) and
[LICENSE-APACHE](https://github.com/Tarvorix/burn_xlstm/blob/main/LICENSE-APACHE). Distributed port code is
subject to both sets of applicable terms (`MIT AND Apache-2.0`).
