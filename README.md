# VQ-bench: a composable vector quantization framework

See [vq-bench.com](https://www.vq-bench.com) for current benchmarks.

## About 

VQ-bench is an open benchmark for vector quantization (VQ) algorithms. It implements a small set of algorithmic primitives (in Rust) and builds quantizers by composing those primitives into pipelines. It then measures how each quantizer trades off **size** (bits per dimension) against **quality** (reconstruction error, score error, recall, etc.) across several high-dimensional embedding datasets, and lets you explore the trade-offs directly.


## Motivation

Vector quantization is an old problem, but it is now central to modern AI:
vector databases use it for large-scale approximate nearest neighbor search,
LLM serving uses it to compress model weights, and growing context windows make
KV-cache compression a memory bottleneck. The result is a rapid increase in quantization papers making strong, hard-to-verify claims. These papers report on different datasets, measure different metrics, run on different hardware, and account for resources differently. VQ-bench aims to provide a single, fair, reproducible benchmark where quantizers are measured side by side.

## Composability

Most quantizers are built from a small, shared set of primitives —
**conditioners** (centering, random rotation, whitening, etc.), **rounders** (integer
casts, k-means codebooks, etc.), and **splitters / routers**.
VQ-bench expresses these primtives through a single interface, so a quantizer is just a pipeline of primitives:

```
PQ      = split(segment).kmeans(lloyd)
SimHash = random_rotate(jl).cast(hamming)
RaBitQ  = adjust(center).adjust(normalize).random_rotate(hadamard).cast(int,1)
```

Because pipelines compose freely, VQ-bench makes it is easy to understand the relationship between different quantizers, create variations on existing quantizers, and invent new quantizers altogether.

VQ-bench is maintained by Amir Ingber, Edo Liberty ([Pinecone](https://pinecone.io/)), and Ashwin Padaki (University of Pennsylvania). The code will become open source in late July 2026.
