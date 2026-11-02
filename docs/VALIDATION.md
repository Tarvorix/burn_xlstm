# Validation record

Validated locally on 2026-11-02 with Burn 0.22.1, macOS CPU/Flex and WGPU
float32. The default suite passes **24 integration tests and 2 doctests**. The
separate GPU test passes the original stack and the complete shape sweep.
This is strong evidence for the tested implementation and configurations; finite
tests cannot prove universal correctness or convergence on arbitrary tasks.

## Reference parity

Fixtures execute the unmodified NX-AI source at
[`ab22eadbd245f293dd8dec38ed29963d73758a12`](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12).
The generator verifies both the commit and a clean source tree. Regeneration
was checked byte for byte. Python dependency versions are pinned in
[`tools/requirements-reference.txt`](../tools/requirements-reference.txt).
Normal Cargo tests consume the checked-in JSON and require no Python.

- Raw sLSTM: outputs, all four final states, and all gate derivatives.
- Raw mLSTM: parallel/recurrent outputs, all three final states, and derivatives
  for queries, keys, values, input gates, and forget gates.
- The original m–s–m fixture plus **12 independently seeded configurations**:
  scalar-only, matrix-only, and mixed stacks; depths 1–3; batches 1–3; widths
  4, 6, 8, 12, 16; heads 1–4; sequence lengths 1–31; projection group widths
  1, 2, 3, 4, 6, 8; convolution widths 0, 2, 3, 4, 7.
- Every sweep case compares full-sequence and chunked outputs, input gradients,
  and **every learned parameter gradient**, on both CPU and GPU. The sweep
  contains 26,398 parameter coordinates across 305 reference tensors.
  All parameters are randomized, including the normally zero recurrent/gate
  weights, to expose missing recurrence and incorrect gate/layout mappings.

The normalized error is `abs(actual - expected) / (1 + abs(expected))`.
Tolerances are `2e-6` for raw outputs/states, `2e-5` for raw gradients, `1e-5`
for stack outputs, and `8e-5` for stack input/parameter gradients. No tolerance
was increased to make the shape sweep pass.

The pinned reference has convolution edge bugs for mLSTM kernel zero and kernel
one. Burn supports these cases; separate streaming tests cover them, rather
than treating broken reference outputs as expected behavior.

## Independent mathematics and state behavior

[`tests/numerics.rs`](../tests/numerics.rs) implements direct, unstabilized paper
equations using Rust f64 scalars and arrays. Moderate inputs avoid overflow.
Three deterministic input sets compare outputs and **612 derivative coordinates**
against central differences with step `1e-5`. These oracles use neither Burn
operations nor PyTorch autodiff. For this independent mLSTM comparison, the
Burn epsilon is `1e-12`, making it negligible against the paper's unit floor;
reference parity separately tests the production `1e-6` epsilon.

Additional checks cover:

- Matrix parallel/recurrent agreement at lengths 127, 512, and **2,048**, and
  gradient agreement at length 257. Float32 cumulative-sum rounding is covered
  by a declared `2e-4` normalized tolerance for these long tests.
- A 257-token mixed stack with irregular chunks, prefill and continuation;
  all memory, normalizer, log-scale, hidden, and convolution-history tensors.
- 1,024 scalar updates with extreme gates, gate logits of ±1,000, and
  cancellation in matrix retrieval; outputs and tested gradients stay finite.
- Causality, independent batch/head lanes, convolution shorter and longer than
  the sequence, and dropout being disabled by `Module::valid()`.
- Checkpoint round trips, state detachment, configuration validation, invalid
  state counts, one-coordinate scalar heads, and language-model streaming.

## Learning on unseen sequences

[`tests/learning.rs`](../tests/learning.rs) trains delayed-token copy using fresh
random batches each optimizer step. The target is the preceding token, with a
zero target at sequence start. **Convolution is disabled**, so copying a previous
token requires recurrent memory.

Settings: width 16, two heads, mLSTM inner width 32, feedforward width 24,
model initialization seed 1729, Adam learning rate 0.003, 400 updates, batch 16,
training length 12, vocabulary size four. The matrix-only model has two mLSTM
blocks; the scalar-only model has one sLSTM; the mixed model has mLSTM then sLSTM.

Two final evaluation seeds, `a66d8c513390ef27` and `509bf17cd3a82261`, were reserved
until the training configuration was fixed. Both generate 64 sequences at each
length, 12 and 24, giving **4,608 evaluated tokens per model**. The length-12
sequences were checked for overlap with all 6,400 training sequences: none.
Length 24 also tests continuation beyond the training length.

| Model | Initial loss, first holdout | Final loss range across four evaluations | Accuracy in each evaluation |
| --- | ---: | ---: | ---: |
| sLSTM | 1.786598 | 0.001484–0.001499 | 100% |
| mLSTM–mLSTM | 1.651991 | 0.001205–0.001313 | 100% |
| mLSTM–sLSTM | 1.524220 | 0.000998–0.001079 | 100% |

The automated threshold is accuracy above 95% and loss below 15% of the initial
loss, independently for every evaluation. These are task-specific convergence
checks, not a language-model quality benchmark. During development, a single
mLSTM block with learning rate 0.01 did not reliably reach that threshold;
training behavior depends on architecture and optimizer settings even when
numerical parity passes. The final configuration uses two matrix blocks and
the lower rate, without changing the xLSTM equations or relaxing the threshold.

## Reproduce

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --test learning -- --nocapture
cargo test --locked --features wgpu --test parity wgpu_stack -- --ignored --nocapture
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
cargo check --locked --no-default-features
cargo check --locked --no-default-features --features wgpu
cargo +1.95.0 check --locked
```

All of the above passed locally. The GPU command needs a working WGPU device;
CPU tests run by default. GitHub Actions runs the default suite, linting,
documentation, and feature builds on Ubuntu and macOS, plus a Rust 1.95 MSRV
build. See the repository's Actions tab for the status of the exact pushed commit.

To regenerate the fixtures in an isolated Python environment:

```sh
python -m pip install -r tools/requirements-reference.txt
git clone https://github.com/NX-AI/xlstm.git /tmp/xlstm-reference
git -C /tmp/xlstm-reference checkout ab22eadbd245f293dd8dec38ed29963d73758a12
python tools/reference_fixtures.py /tmp/xlstm-reference
git diff --exit-code -- tests/fixtures
```

## Remaining scope

Windows, other GPU vendors/backends, mixed precision, large-model training,
throughput, and distributed execution have not been validated. These tests do
not validate NX-AI's optimized CUDA/Triton kernels or xLSTM-large, which this
crate does not implement. Long parallel mLSTM still uses quadratic storage;
stateful execution provides bounded recurrent state for inference.
