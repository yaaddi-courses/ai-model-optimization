# INT4, INT32 and FP64: the formats at both extremes

**Covers:** INT4, 4-bit weights, INT32, FP64

## Beyond the common formats {#overview}
FP32, FP16 and INT8 cover most everyday work, but you will meet formats on either side of them. Some are far smaller and squeeze a model into very little memory. Others are far larger and are rarely worth their cost in a model. Knowing where each one sits stops you from confusing them, which is a classic trap in exam questions and in real planning.

## INT4 {#int4}
**INT4** is a 4-bit whole-number format. Four bits give only 16 different patterns, so the values run from −8 to 7. Two INT4 values fit in a single byte.

Such a coarse grid cannot store a weight exactly, so INT4 is always used together with a scale that stretches the 16 steps over the real range of the weights. It is used to shrink the weights of large models to a quarter of their size at 16 bits.

## 4-bit weights {#4-bit-weights}
**4-bit weights** is the name for a model whose weights are each stored in 4 bits. This is usually INT4 or a similar 4-bit code. The memory saving is large: a model with 7 billion weights needs about 3.5 GB for its weights, compared with 14 GB at 16 bits.

The price is coarser numbers. Careful methods keep the accuracy loss small, and many large models run well at 4 bits. Small models tend to suffer more, because they have less spare capacity to absorb the rounding.

## INT32 {#int32}
**INT32** is a 32-bit whole-number format, holding values up to about 2.1 billion. It is not a smaller version of anything. It uses 32 bits per value, twice as many as FP16. You will see it inside hardware as a place to add up many small integer products without running out of room, and in indexing and counting code, but not as a way to store weights.

## FP64 {#fp64}
**FP64** is double-precision floating point, using 64 bits per value, twice the memory of FP32. It keeps about 15 to 16 significant digits. Scientific simulations need that detail. Machine-learning models almost never do, since their predictions are not sensitive to the 10th digit. Using FP64 for a model would double its size and slow it down for no gain in quality.

## A worked example {#example}
The table of memory for the weights of a 7-billion-weight model shows how the sizes compare.

```python
weights = 7_000_000_000
bits = {"FP64": 64, "INT32": 32, "FP32": 32, "FP16": 16, "INT8": 8, "INT4": 4}

for name, b in bits.items():
    gigabytes = weights * b / 8 / 1e9
    print(f"{name:5} {gigabytes:6.1f} GB")
# FP64   56.0 GB
# INT32  28.0 GB
# FP32   28.0 GB
# FP16   14.0 GB
# INT8    7.0 GB
# INT4    3.5 GB
```

How to read it:

1. Divide the bits by 8 to get bytes per weight, then multiply by the number of weights.
2. INT32 and FP32 take the same 28 GB, because both use 32 bits.
3. INT4 needs one-eighth of FP32 and half of INT8.
4. The smallest rows are not free: each step down trades away detail.

## A common mistake {#mistake}
Mixing up a number's type with its size. A format's name tells you the bits per value, not whether it is a decimal or whole-number format. INT32 is whole numbers in 32 bits, while FP32 is decimals in 32 bits, so INT32 is not a "smaller" or more compact choice than FP16.

## Key points {#key-points}
- INT4 stores 16 values (−8 to 7), so it always needs a scale to be useful.
- 4-bit weights cut a model's weight memory to a quarter of the 16-bit size.
- INT32 uses 32 bits per value and serves counting and accumulation, not weight storage.
- FP64 doubles FP32's memory and is not worth it for model weights.
