# Batching: keeping the hardware busy

**Covers:** Batching, Utilization, GPU utilization, Dynamic batching, Padding

## One at a time wastes the chip {#overview}
A GPU is built to do thousands of operations at the same moment. If you give it one request at a time, most of that capacity sits idle while it reads the weights for a single small job. Think of a bus that drives across town with one passenger. The better plan is to hand the chip several requests together, so one reading of the weights serves all of them. This is the most widely used way to raise throughput, and in this article we look at how it works and where it costs you.

## Batching {#batching}
**Batching** means grouping several inputs and processing them in one pass. The weights are read once, and every input in the group is multiplied with them together. Throughput rises, because the cost of reading is shared. There is a trade-off, though. Each request may wait for the group to fill, and the group takes a little longer to compute than a single request would.

## Utilization {#utilization}
How do you know whether a chip is being used well? **Utilization** is the share of a chip's capacity that is doing useful work. A chip that is busy 30% of the time, or that uses 30% of its arithmetic units, has a utilization of 30%. Low utilization is a sign that you are paying for capacity you do not use.

## GPU utilization {#gpu-utilization}
**GPU utilization** is the same measure applied to a GPU. Since GPUs are expensive, it is one of the first numbers people check. A low value on a loaded server usually means requests are being handled too one by one, and that batching or better scheduling could help. A high value is a good sign, but it does not prove everything is perfect, because a busy chip can still be doing wasteful work.

## Dynamic batching {#dynamic-batching}
Requests do not arrive in tidy groups. They show up one by one, at random moments. So who forms the batch? In **dynamic batching**, the server collects incoming requests for a short window of time, up to a maximum size, and then runs them together. You set the window and the size. A short window keeps latency low, and a long one makes bigger groups.

## Padding {#padding}
Here is a detail that surprises people. Inputs in a group must have the same shape, but real inputs have different lengths. **Padding** adds dummy filler values to the shorter ones until they all match the longest. The chip does arithmetic on the filler even though it carries no information, so padding costs compute and gives nothing back. A group that mixes very short and very long inputs wastes a lot.

## A worked example {#example}
Let's do two small experiments. First, we see how a 10 ms window groups six requests when the maximum group is four. Then we measure the cost of padding.

```python
def group_requests(arrivals, window, max_size):
    groups = []
    i = 0
    while i < len(arrivals):
        start = arrivals[i]
        group = [start]
        i += 1
        while i < len(arrivals) and len(group) < max_size and arrivals[i] < start + window:
            group.append(arrivals[i])
            i += 1
        groups.append(group)
    return groups

print(group_requests([0, 2, 3, 15, 16, 40], window=10, max_size=4))
# [[0, 2, 3], [15, 16], [40]]

lengths = [5, 9, 12]
padded = max(lengths) * len(lengths)
print(sum(lengths), padded, round(sum(lengths) / padded, 2))   # 26 36 0.72
```

Walking through it:

1. The first request starts a window from 0 to 10 ms, so the requests at 2 and 3 ms join it.
2. The request at 15 ms starts a new window and picks up the one at 16 ms.
3. The request at 40 ms is alone, and it waits out its window for nothing.
4. In the padding part, three inputs of lengths 5, 9 and 12 are padded to 12 each. Only 26 of 36 positions hold real data, so 72% of the work is useful and 28% is padding.

## A common mistake {#mistake}
A common slip is making the batch as large as possible. It feels efficient, but past a point the extra requests mostly add waiting. Choose the window and the size from your latency target, not from the largest number your memory allows.

## Key points {#key-points}
- Batching shares one reading of the weights across several requests, which raises throughput.
- Utilization tells you how much of the chip's capacity does useful work.
- Dynamic batching collects requests for a short window up to a size limit.
- Padding makes lengths match but wastes compute, so group similar lengths when you can.
