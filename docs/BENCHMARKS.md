# mLSTM kernel benchmarks

Measured on 2026-11-02 on an Apple M5 with 32 GiB unified memory, macOS,
Burn 0.22.1, Rust release builds, and float32. CPU uses Flex; GPU uses WGPU/Metal.
The reference inputs have batch 2, four heads, head dimension 16, and chunk
size 64. Random inputs use seed 123 and moderate gates.

Each result is the median of three measured samples after one warmup. Device
synchronization and a scalar result/gradient checksum readback ensure work
finishes inside the timing interval. Forward/backward checksums consume all five
input gradients. These timings exclude model projections, sLSTM, normalization,
convolution, vocabulary output, an optimizer, and data loading. They are kernel
microbenchmarks, not end-to-end training measurements or peak-memory measurements.

## CPU / Flex

Milliseconds; lower is faster. Raw results: [m5-cpu.csv](benchmarks/m5-cpu.csv).

| Tokens | Phase | Full parallel | Chunkwise (64) | Token recurrent |
| ---: | --- | ---: | ---: | ---: |
| 128 | forward | 0.995 | 0.669 | 1.290 |
| 128 | forward + backward | 1.351 | 1.052 | 13.419 |
| 512 | forward | 8.098 | 1.353 | 5.548 |
| 512 | forward + backward | 23.293 | 4.970 | 121.649 |
| 2048 | forward | 140.730 | 5.553 | 24.509 |
| 2048 | forward + backward | 403.847 | 24.095 | 1110.560 |

## GPU / WGPU

Milliseconds; lower is faster. Raw results: [m5-wgpu.csv](benchmarks/m5-wgpu.csv).

| Tokens | Phase | Full parallel | Chunkwise (64) | Token recurrent |
| ---: | --- | ---: | ---: | ---: |
| 128 | forward | 1.071 | 6.732 | 37.629 |
| 128 | forward + backward | 6.679 | 7.812 | 248.003 |
| 512 | forward | 5.146 | 5.311 | 176.519 |
| 512 | forward + backward | 11.405 | 12.467 | 1794.534 |
| 2048 | forward | 47.362 | 8.960 | 616.768 |
| 2048 | forward + backward | 157.972 | 37.386 | 15203.444 |

## Interpretation

At 2,048 tokens, chunkwise execution was about 25.3× faster forward and 16.8×
faster forward/backward on CPU, and 5.3× / 4.2× faster on GPU, relative to the
full parallel path. At 128 and 512 tokens the GPU's full parallel path was
faster. Chunk size 64 is one tested setting, not a universally optimal choice.
Use representative workloads to choose a path and chunk size.

The mathematical interaction-storage reduction is separate from timing: at
2,048 tokens, one full `[2, 4, 2048, 2048]` float32 interaction tensor is 128 MiB.
One 64-token chunk is 128 KiB; retaining such a tensor for all 32 chunks is
4 MiB, a 32× reduction for this class of intermediates. Other intermediates,
chunk-boundary matrices, activations, gradients, and allocator caches still
consume memory. These figures are not measured total peak memory.

Padding/packed execution currently uses token recurrence so arbitrary mask holes
and sequence boundaries are exact. It improves data handling and storage use,
but packing does not automatically improve throughput. Uniform-length batches
can use the faster parallel/chunkwise APIs. More fused GPU work is a possible
future optimization; these results do not establish performance on NVIDIA/AMD
hardware or justify claiming that custom CUDA/Triton kernels have been ported.

## Reproduce

```sh
cargo run --release --example benchmark
cargo run --release --features wgpu --example benchmark -- --wgpu
```

Run benchmarks without other training, compilation, or GPU work competing for
resources. Repeated runs and different machines can produce different timings.
