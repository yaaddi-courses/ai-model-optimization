# Memory-bound or compute-bound: what limits your model?

**Covers:** Memory-bound, Compute-bound, Bandwidth

## Two kinds of slow {#overview}
Think of a kitchen. A cook can be slow because chopping takes too long, or because the ingredients keep arriving late from the storeroom. A chip can be slow for the same two reasons. It may be short of arithmetic speed, so the multiplications themselves take too long. Or it may be short of data, so the arithmetic units sit idle waiting for numbers to arrive from memory. The remedies are opposite. A chip with faster arithmetic does nothing for a job that is waiting on memory. So the first question is always which case you are in.

## Bandwidth {#bandwidth}
Let's start with the memory side. Memory **bandwidth** is how many bytes per second a chip can read from or write to its memory. A modern data-centre GPU may reach about 1,000 to 3,000 gigabytes per second, while the memory in an ordinary computer manages a few tens of gigabytes per second. Bandwidth is a fixed property of the hardware, like the width of a pipe.

## Compute-bound {#compute-bound}
Work is **compute-bound** when the chip's arithmetic speed is the limit. Each number that is loaded gets used many times, so the memory keeps up and the multipliers are the busy part. Large matrix multiplications, where the same values are combined over and over, are the classic case. Training, and the first pass over a long input, tend to be compute-bound.

## Memory-bound {#memory-bound}
Work is **memory-bound** when the speed of reading data is the limit. The chip could do the arithmetic far faster than the data arrives. Writing out a reply one piece at a time is the standard example. For every new piece, the model must read all of its weights from memory, yet it does only about two multiplications with each. The arithmetic is quick, and the reading is the slow part.

## How to tell them apart {#telling-apart}
You can check with two times. The first is the time to read the data once at the chip's bandwidth. The second is the time to do all the arithmetic at the chip's speed. Whichever is larger is the limit.

## A worked example {#example}
Let's try it. A model has 7 billion weights stored in 2 bytes each, and it produces one piece of text at a time. The chip has 1,000 GB/s of bandwidth and 100 trillion operations per second. These round figures are for illustration.

```python
weights = 7e9
bytes_per_weight = 2
bandwidth = 1000e9            # bytes per second
compute_speed = 100e12        # operations per second

read_time = weights * bytes_per_weight / bandwidth
math_time = 2 * weights / compute_speed

print(round(read_time * 1000, 2))   # 14.0 milliseconds to read the weights once
print(round(math_time * 1000, 2))   # 0.14 milliseconds of arithmetic
print(round(1 / read_time, 1))      # 71.4 pieces per second at best
```

Reading the output:

1. Reading 14 GB of weights at 1,000 GB/s takes 14 ms.
2. The arithmetic, two operations per weight, takes only 0.14 ms.
3. Reading is 100 times slower than the math, so the job is firmly memory-bound.
4. The best possible rate is about 71 pieces per second, however fast the multipliers are.

Doubling the arithmetic speed would change nothing here. Halving the bytes per weight would double the rate, and that is why shrinking the weights speeds up text generation.

## A common mistake {#mistake}
A common slip is buying a faster chip without checking what limits you. If your workload is memory-bound, look for more bandwidth, smaller weights, or ways to reuse each loaded weight for more work, such as handling several requests together.

## Key points {#key-points}
- Bandwidth is bytes per second to and from memory, and it is a fixed property of the hardware.
- Compute-bound work waits for arithmetic, and memory-bound work waits for data.
- Producing text one piece at a time is usually memory-bound.
- Compare read time with arithmetic time to find the limit, then pick the matching fix.
