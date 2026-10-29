# Operator fusion and kernels: doing more per trip to memory

**Covers:** Operator fusion, Kernel, ReLU, CUDA, NVIDIA

## The hidden cost of small steps {#overview}
A model is a chain of small operations: multiply, add, apply a function, and so on. On a GPU each of these is a separate piece of work, and between them the intermediate result is written out to memory and read back in. Reading and writing memory is slow compared with the arithmetic itself. A chain of ten tiny steps can spend most of its time moving data and very little time computing. Operator fusion removes that waste.

## NVIDIA and CUDA {#cuda}
**NVIDIA** is the company that makes most of the GPUs used to train and run AI models. **CUDA** is the software platform NVIDIA provides so that ordinary programs can run on its GPUs. Almost all machine-learning software on NVIDIA hardware sits on top of CUDA, which is why you meet the name constantly in error messages and install guides. A tool that says it needs CUDA needs an NVIDIA GPU.

## Kernel {#kernel}
A **kernel** is a small program that runs on a GPU to perform one operation, such as multiplying matrices or adding two lists of numbers. It is launched across thousands of cores at once. The word has nothing to do with the operating-system kernel. Each kernel launch has a fixed overhead, and each one reads its inputs from memory and writes its output back.

## ReLU {#relu}
**ReLU**, short for rectified linear unit, is a simple function used between layers: it returns zero for negative inputs and the input itself otherwise. In code it is `max(0, x)`. It is cheap to compute, which makes it a good example of an operation whose cost is almost entirely the memory traffic around it.

## Operator fusion {#operator-fusion}
**Operator fusion** merges adjacent operations into a single kernel. The intermediate result stays in fast on-chip storage instead of going out to memory and back. For a chain such as "add the bias, then apply ReLU", a fused kernel reads the inputs once, does both steps, and writes the final answer once.

## A worked example {#example}
Adding two tensors of one million numbers each, 2 bytes per number, and then applying ReLU. Count the memory traffic with and without fusion.

```python
elements = 1_000_000
bytes_per_number = 2
tensor_mb = elements * bytes_per_number / 1e6          # 2.0 MB per tensor

# Separate kernels: add, then relu
add_traffic = 2 * tensor_mb + tensor_mb                # read a and b, write c
relu_traffic = tensor_mb + tensor_mb                   # read c, write d
separate = add_traffic + relu_traffic

# One fused kernel: read a and b, write d
fused = 2 * tensor_mb + tensor_mb

print(separate, fused)                                 # 10.0 6.0
print(round(1 - fused / separate, 2))                  # 0.4

def relu(x):
    return max(0, x)

print([relu(v) for v in [-2, 0, 3]])                   # [0, 0, 3]
```

Reading it:

1. The add kernel reads two 2 MB tensors and writes a 2 MB result: 6 MB of traffic.
2. The ReLU kernel then reads that 2 MB result and writes another 2 MB: 4 MB.
3. Separately, the total is 10 MB. Fused, the temporary result is never written, so the total is 6 MB.
4. That is 40% less memory traffic, and since these operations are limited by memory, close to 40% less time.

The arithmetic did not change at all. Only the trips to memory did.

## A common mistake {#mistake}
Expecting fusion to speed up everything. It helps most for chains of cheap operations that are limited by memory. A large matrix multiplication already spends its time computing, so fusing around it gains little.

## Key points {#key-points}
- Each separate GPU operation writes its result to memory and the next one reads it back.
- A kernel is one such operation launched on the GPU.
- Operator fusion merges neighbouring operations into one kernel, cutting memory traffic.
- ReLU, CUDA and NVIDIA are the function, platform and company you meet around this topic.
