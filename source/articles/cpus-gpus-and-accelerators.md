# CPUs, GPUs and other accelerators: choosing the chip

**Covers:** CPU, GPU, VRAM, NPU, TPU, Accelerator

## Why the chip matters {#overview}
The same model can answer in a fraction of a second on one chip and take many seconds on another. Why such a gap? A model is made of a huge number of simple multiplications, and chips differ greatly in how well they handle that kind of work. So choosing the chip is one of the biggest levers you have on speed and cost. Let's go through the main kinds and what each one is good at.

## CPU {#cpu}
You are probably using one right now. A **CPU**, or central processing unit, has a few powerful, flexible cores. It is excellent at logic with many branches, such as "if this, then that", and at running an operating system and ordinary programs. It can run a model too, but with only a handful of cores doing the multiplication work, it is slow for big models.

## GPU {#gpu}
A **GPU**, or graphics processing unit, takes the opposite approach. It has thousands of small cores that all do the same simple operation at once. That suits the heavy multiplication in a model very well. A GPU also has its own fast memory. It is the usual choice for training models and for serving them.

## VRAM {#vram}
That own memory has a name: **VRAM**, the GPU's dedicated memory. For fast answers, a model's weights generally need to fit in VRAM. A GPU with 24 GB of VRAM cannot hold a 28 GB model without help, however fast its cores are. When a model does not fit, you have three options: shrink it, split it across more than one GPU, or accept slow transfers from regular memory.

## NPU {#npu}
An **NPU**, or neural processing unit, is a low-power chip built for neural-network arithmetic. You find it in phones and laptops, where battery life matters. It is not as fast as a large GPU, but it uses much less energy, which makes it a good home for small models on a device.

## TPU {#tpu}
A **TPU**, or tensor processing unit, is Google's custom chip for tensor operations in machine learning. It is mostly available through Google's cloud. It is designed around large matrix multiplications and works with software such as JAX and TensorFlow.

## Accelerator {#accelerator}
**Accelerator** is the general name for a chip specialised for a workload such as neural networks. GPUs, TPUs and NPUs are all accelerators. The word separates them from the CPU, which can do everything but is not specialised for any one thing.

## A worked example {#example}
A single large matrix multiplication is a good test of the difference. The speeds below are round figures chosen for illustration. They are not measurements of real chips.

```python
size = 4096
operations = 2 * size ** 3             # multiply-adds in a size x size by size x size product
print(round(operations / 1e9, 1))      # 137.4 billion operations

cpu_speed = 0.5e12                     # assume 0.5 trillion operations per second
gpu_speed = 100e12                     # assume 100 trillion operations per second

print(round(operations / cpu_speed, 3))            # 0.275 seconds
print(round(operations / gpu_speed * 1000, 2))     # 1.37 milliseconds
```

What the numbers tell us:

1. Multiplying two 4096 by 4096 matrices needs about 137 billion operations.
2. On the assumed CPU, that takes about 0.275 seconds.
3. On the assumed GPU, it takes about 1.4 milliseconds, around 200 times faster.
4. A real model repeats such products thousands of times, so the gap becomes the difference between waiting and not waiting.

## A common mistake {#mistake}
A common slip is judging a chip by its speed alone. A fast accelerator with too little VRAM cannot run a big model at all. Check the memory first, then the speed, then the price.

## Key points {#key-points}
- A CPU has a few flexible cores, and a GPU has thousands of simple ones suited to model arithmetic.
- VRAM is the GPU's own memory, and the weights usually need to fit in it.
- NPUs are low-power chips for devices, and TPUs are Google's tensor chips.
- Accelerator is the general word for chips specialised for neural-network work.
