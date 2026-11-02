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

`forward` starts a fresh sequence. By default it uses full parallel mLSTM;
`.with_mlstm_chunk_size(Some(64))` selects parallel chunks of at most 64 tokens.
`forward_with_state` accepts arbitrary nonempty chunks and returns continuation
memory, using the configured chunkwise path or recurrent steps.
`step` requires time length one. Pass `None` to reset a sequence. State must match
the model, batch, device, and dtype; it retains its backward graph until detached.

Use `Device::flex().autodiff()` for training. Burn's `Module::valid()` produces an
inference model with dropout disabled; `Module::train()` enables training again.
Sequence/stream comparisons require dropout to be disabled. Full and recurrent
mLSTM can differ slightly because their log stabilizers and epsilon scales differ.

## Variable-length batches

`LmBatch::padded` creates aligned next-token inputs and targets from independent
sequences. `LmBatch::packed` combines several sequences into a fixed-width row;
it resets memory and convolution at each boundary and never creates a target
that crosses from one original sequence into another.

```rust
use burn::tensor::Device;
use burn_xlstm::{batch::LmBatch, lm::XlstmLmConfig, stack::{BlockKind, XlstmConfig}};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let device = Device::flex().autodiff();
let config = XlstmConfig::new(16, vec![BlockKind::Mlstm, BlockKind::Slstm])
    .with_num_heads(2).with_mlstm_chunk_size(Some(64));
let model = XlstmLmConfig::new(32, config).init(&device)?;
let sequences = vec![vec![1, 2, 3, 4], vec![5, 6, 7]];
let batch = LmBatch::padded(&sequences, &device)?;
// Alternatively: LmBatch::packed(&sequences, 8, &device)?;
let (logits, state) = model.forward_masked(batch.inputs.clone(), &batch.mask, None);
let loss = batch.loss(logits);
let gradients = loss.backward();
# let _ = (state, gradients);
# Ok(())
# }
```

For continuous features or custom targets, use `SequenceMask`,
`Xlstm::forward_masked`, and `batch::masked_cross_entropy`. `valid = true` means
consume the token; false positions output zeros and preserve **all** recurrent
and convolution state. `reset = true` clears a lane before its valid token;
reset flags at invalid positions are ignored. An entirely masked loss is finite
zero. Masked token IDs and targets are sanitized before embedding/loss evaluation.

The masked path runs recurrently to support arbitrary mask holes and packed
boundaries. For uniform-length batches, use ordinary `forward` with parallel
chunks for higher throughput. Dropout must be disabled for numerical comparisons
between packed, padded, and separate-sequence processing.

## Parallel chunkwise execution

Set `.with_mlstm_chunk_size(Some(64))` on `XlstmConfig`, or `.with_chunk_size(Some(64))`
on `MlstmConfig`. Matrix products evaluate tokens within each chunk in parallel;
recurrent matrix memory connects chunks, including their gradients. The last
chunk can be shorter. Existing state can be supplied to `forward_with_state`.

This reduces the interaction intermediates from O(time²) to O(time × `chunk_size`)
during training. It also retains chunk-boundary matrix states and ordinary model
activations; this is not a claim of constant total training memory. The default
`None` retains the original full parallel path for reference comparisons.

Run synchronized CPU/GPU kernel benchmarks with:

```sh
cargo run --release --example benchmark
cargo run --release --features wgpu --example benchmark -- --wgpu
```

These measure forward and forward/backward execution at lengths 128, 512, and
2,048, with a warmup and the median of three samples. Timings include scalar
readback to ensure GPU work executes. They compare full parallel, chunkwise,
and token-recurrent mLSTM. See the [benchmark record](https://github.com/Tarvorix/burn_xlstm/blob/main/docs/BENCHMARKS.md)
for measurements; these are not end-to-end language-model timings.

## Implemented structure

| Component | Behavior |
| --- | --- |
| sLSTM | Exponential input gate, sigmoid forget gate, stabilized scalar memory/normalizer, sigmoid output gate, headwise recurrent memory mixing |
| sLSTM layer | Causal depthwise convolution + `SiLU` for input/forget projections; unconvolved candidate/output projections; dropout then head normalization |
| sLSTM block | Pre-LayerNorm memory residual, followed by a separate pre-LayerNorm GELU-gated feedforward residual |
| mLSTM | Stabilized key/value covariance update, scaled keys, absolute normalizer with floor; recurrent, causal parallel, and parallel chunkwise execution |
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

The default suite passes 31 integration tests and 3 doctests. Scalar-only,
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

This crate implements the defining components of the original 2024 xLSTM:
stabilized exponential gating, scalar memory with recurrent mixing (sLSTM),
matrix memory with covariance updates (mLSTM), and normalized residual blocks.
It is an xLSTM implementation; the following limits describe execution efficiency,
optional features, and compatibility with other model variants.

| Feature | Current support and when it matters |
| --- | --- |
| Specialized CUDA/Triton kernels | Computation uses portable Burn tensor operations, including GPU execution through WGPU. NX-AI's specialized kernels are not ported; their absence affects performance, not the defining equations. |
| Parallel chunkwise mLSTM | Available through `mlstm_chunk_size`; parallel work inside chunks carries differentiable memory between chunks. The full parallel and token-recurrent paths remain available. |
| xLSTM-large / pretrained 7B | NX-AI's later updated architecture and pretrained weights are not implemented. They are separate from the original architecture supported here. |
| Packed/ragged sequences | `LmBatch::padded` and `LmBatch::packed` batch variable-length token sequences. Storage remains rectangular; explicit masks and resets preserve independent sequences. |
| Padding masks | `forward_masked` skips memory/convolution updates and zeros invalid outputs. `masked_cross_entropy` ignores invalid targets. |
| Weight tying | Token embeddings and vocabulary output weights are independent. Sharing them is an optional parameter-saving feature. |
| General `PyTorch` checkpoint import | Burn checkpoints are supported. Loading arbitrary NX-AI checkpoints would require a converter and matching architecture/configuration; the test fixture loader is not a general importer. |

Bias-free projections and GELU feedforwards match the selected reference
configuration. Float32 CPU and WGPU numerical parity are tested. Mixed-precision
training has not been validated. Kernel benchmarks are provided; end-to-end
large-model throughput has not been certified. Stateful
inference retains bounded recurrent memory; sLSTM necessarily runs recurrently.

NX-AI documents its [updated xLSTM Large architecture](https://github.com/NX-AI/xlstm#xlstm-7b-a-recurrent-llm-for-fast-and-efficient-inference)
and [optimized mLSTM kernels](https://github.com/NX-AI/mlstm_kernels) separately.

Two numerical edge cases are handled deliberately: empty sLSTM lanes initialize
their scale independently, and readout avoids directly evaluating `exp(-m)` when
`m` is very negative. One-dimensional scalar heads use the finite left endpoint
of the reference's forget-bias schedule.

## Publishing

The initial crates.io release is prepared as `burn-xlstm` version `0.1.0`.
See the [publishing guide](https://github.com/Tarvorix/burn_xlstm/blob/main/docs/PUBLISHING.md)
for account setup, package verification, and the release commands.

## License

Licensed under the [Apache License, Version 2.0](https://github.com/Tarvorix/burn_xlstm/blob/main/LICENSE),
matching the [NX-AI/xlstm reference implementation](https://github.com/NX-AI/xlstm/blob/ab22eadbd245f293dd8dec38ed29963d73758a12/LICENSE).
Upstream attribution and third-party notices are preserved in
[NOTICE](https://github.com/Tarvorix/burn_xlstm/blob/main/NOTICE).
