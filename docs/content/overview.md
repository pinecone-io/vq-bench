# Overview

**VQ-bench** is an open-source benchmark and framework for vector quantization. The purpose of this documentation is to explain the core motivation and design principles behind VQ-bench and to provide detailed guidelines for contributing to the project. 

## Motivation

Vector quantization is an old problem, but it is now central to modern AI infrastructure. Vector databases use it for large-scale approximate nearest neighbor search, LLM servers use it to compress model weights, and KV caching techniques use it to scale context windows. As a result, vector quantization is experiencing a surge of engineering and research effort, with hundreds of papers being published on the topic in recent years.

The renewed interest in vector quantization has made it increasingly difficult to keep track of the state of the art. Published results are often hard to compare: they use different datasets, measure different quality metrics, run on different hardware, and account for resource consumption differently. Incomplete credit attribution makes it harder to distinguish novel ideas from variations of existing algorithms.

The goal of VQ-bench is to bring some order to the chaos. We thoroughly evaluate a large set of quantizers using several metrics and present the results openly. Our framework distills quantizers as pipelines of a shared set of primitives, making it easy to understand the relationship between different quantizers, create variations of existing ones, and invent new quantizers altogether.

## Design Principles

The core design principles of VQ-bench are **standardization** and **composability**.

### Standardization

VQ-bench treats every **quantizer** as a black box that must support four operations: 
1. Given a sample of the dataset (and optionally queries), return a fitted **model**.
2. Given the model and the dataset, return per-vector **codes**. The size of the **quantization** is the total size of the model and all per-vector codes.
3. Given dataset codes, return approximate **reconstructions** of the corresponding vectors.
4. Given a query and dataset codes, returns approximate inner-product **scores** for the corresponding vectors.

The job of the harness is to perform the above operations on all quantizers (for a suite of datasets) and use the returned reconstructions and scores to compute several quality metrics. The final [results](index.html) are presented interactively so that different tradeoffs can be explored.

### Composability

In VQ-bench, the default way to build a quantizer is by specifying a pipeline of shared **primitives**. VQ-bench maintains a catalog of primitives (implemented in Rust) which fall into three main classes.
- **Conditioners:** These are transformations applied to a set of vectors to achieve a desired geometric or statistical property. Examples including *centering*, *normalizing*, and *random rotation*.
- **Rounders:** These are mappings from each vector to a fixed or learned codebook. Examples include *integer casting*, *angular casting*, and *k-means rounding*. 
- **Splitters:** These are branching operations which split the dataset into chunks, each handled by downstream primitives. A common example is *segmenting by columns*.

Many vector quantization algorithms can be written as compositions of primitives. For example:
```
SimHash   = center.normalize.random_rotate.cast_hamming
E-RaBitQ  = center.normalize.random_rotate.cast_angular
PQ        = segment_columns.kmeans
```

## Contributing to VQ-bench

First, familiarize yourself with [how to use VQ-bench](docs.html#usage). If you're interested in contributing, check out the guides on [adding a primitive](docs.html#adding-a-primitive) and [adding a quantizer](docs.html#adding-a-quantizer).