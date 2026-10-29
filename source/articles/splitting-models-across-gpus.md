# Splitting a model across GPUs

**Covers:** Parallelism, Sharding, Data parallelism, Tensor parallelism, Pipeline parallelism

## When one GPU is not enough {#overview}
Some models are too big for one GPU's memory. A model with 70 billion weights at 16 bits needs about 140 GB for the weights alone, and the largest single GPUs offer around 80 GB. Other times a model fits, but a single GPU cannot serve enough users. In both cases the answer is to use more than one GPU. There are several ways to divide the work, and each solves a different problem. Let's go through them.

## Parallelism {#parallelism}
**Parallelism** means doing many operations at the same time. It is what makes a GPU fast in the first place, since thousands of cores multiply different numbers at once. In multi-GPU serving, the word covers the different ways of spreading a model or its traffic over several GPUs, which we describe below.

## Sharding {#sharding}
**Sharding** splits a model's weights into pieces, called shards, and stores each piece on a different GPU. A model that is too big for one GPU can then run, because no single GPU has to hold all of it. Sharding is the basic idea behind the two model-splitting methods that follow.

## Data parallelism {#data-parallelism}
**Data parallelism** runs several complete copies of the model on different GPUs, and each copy serves different requests. It adds traffic capacity, since four copies handle roughly four times the requests. It does not help a model that is too big, because every GPU must still hold the whole thing.

## Tensor parallelism {#tensor-parallelism}
**Tensor parallelism** splits each layer's weight matrices across several GPUs. The GPUs work on the same request together, each doing its part of every layer, and they combine their results after each layer. This cuts latency for a single request and lets very large models run. The price is frequent, fast communication between the GPUs, so it works best when they are tightly linked.

## Pipeline parallelism {#pipeline-parallelism}
**Pipeline parallelism** splits a model by layers. The first GPU holds the first block of layers, the second holds the next, and so on. A request flows from one GPU to the next like an assembly line. Communication is much lighter than in tensor parallelism, because only the activations at block boundaries move. But a single request still passes through the stages one after another, so its latency does not improve. Throughput can improve when several requests keep every stage busy.

## A worked example {#example}
Let's use a model with 140 GB of weights and GPUs with 80 GB each.

```python
import math

model_gb = 140
gpu_gb = 80
print(math.ceil(model_gb / gpu_gb))        # 2 GPUs are the minimum to hold the weights

# Tensor parallelism over 2 GPUs: each holds half of every layer
print(model_gb / 2)                         # 70.0 GB of weights per GPU

# Pipeline parallelism over 4 GPUs with 80 layers: each holds a block of layers
layers = 80
gpus = 4
print(layers // gpus, model_gb / gpus)      # 20 35.0

# Data parallelism: 4 full copies need 4 times the GPUs, 2 each
copies = 4
print(copies * math.ceil(model_gb / gpu_gb))   # 8 GPUs for 4 copies
```

Step by step:

1. The weights alone need at least two 80 GB GPUs.
2. With tensor parallelism over two GPUs, each stores 70 GB, which leaves 10 GB per GPU for working memory.
3. With a pipeline over four GPUs, each holds 20 of the 80 layers and 35 GB.
4. To also serve four times the traffic with data parallelism, the model must be split first and then repeated, which takes eight GPUs in this example.

Real deployments combine the methods: split the model across the GPUs of one machine, then copy that machine for capacity.

## A common mistake {#mistake}
A common slip is adding GPUs without checking the links between them. Tensor parallelism across slow connections can run slower than on one GPU, because the time spent communicating exceeds the time saved computing.

## Key points {#key-points}
- Sharding places pieces of a model on different GPUs so a large model can run.
- Data parallelism copies the whole model for more traffic, but not for more size.
- Tensor parallelism splits each layer, which lowers latency but needs fast links.
- Pipeline parallelism splits by layers, which communicates less but does not reduce a single request's latency.
