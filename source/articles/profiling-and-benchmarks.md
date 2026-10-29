# Profiling and benchmarks: measure before you change anything

**Covers:** Profiling, Bottleneck, Benchmark, Warm-up run, p95 latency, p99 latency

## Guessing is expensive {#overview}
The most common way to waste a week is to speed up the wrong thing. Someone sees a slow model, picks the part that looks heavy, rewrites it, and the whole system gets 2% faster because that part was never the problem. The cure is to measure first. This article covers the tools for measuring and the habits that keep the numbers honest.

## Profiling {#profiling}
**Profiling** means measuring how long each part of a program takes, and how much memory it uses, so you can see where the time really goes. A profile might show that loading the input takes 30 ms, the model takes 60 ms and turning the result into text takes 10 ms. Nobody would have guessed the loading step, but there it is.

Run the profiler on a realistic request, not a toy one. A profile of an easy input can point at the wrong part.

## Bottleneck {#bottleneck}
A **bottleneck** is the single slowest step that limits the whole system. Think of a road that narrows to one lane: it does not matter how wide the rest is, because traffic can only move as fast as that lane.

The profile above makes the point. If the model takes 60 of 100 ms, it is the bottleneck. Making the loading step twice as fast saves 15 ms, while making the model twice as fast saves 30 ms. After you fix the bottleneck, another step becomes the slowest, so the loop starts again.

## Benchmark {#benchmark}
A **benchmark** is a repeatable test that measures speed or accuracy under fixed conditions. The word fixed is the point. Use the same inputs, the same hardware and the same settings each time, so that a difference in the result means the model changed and nothing else did.

Without a benchmark, "it feels faster" is an opinion. With one, you can compare version A against version B and put a number on the difference.

## Warm-up run {#warm-up-run}
The first time a model runs, it is slower. Weights are being loaded, caches are empty and the hardware may still be waking up. A **warm-up run** is an untimed run before you start measuring, so these one-time costs do not pollute your numbers. Do a few warm-up runs, throw their times away, then time the real ones.

## p95 latency and p99 latency {#p95-latency}
Averages hide the worst moments. A percentile says what share of requests were faster than a given time. **p95 latency** is the time that 95% of requests beat. Only 1 in 20 is slower. **p99 latency** is the time that 99% of requests beat, so only 1 in 100 is slower.

These are sometimes called tail latencies, and users notice them. A person who gets one slow answer in twenty remembers it.

## A worked example {#example}
Twenty timed requests, in milliseconds, with one very slow one:

```python
import math

latencies = [95, 97, 98, 99, 100, 100, 101, 102, 103, 104,
             105, 106, 107, 108, 109, 110, 112, 115, 120, 900]

def percentile(values, p):
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]

print(sum(latencies) / len(latencies))   # 144.55
print(percentile(latencies, 50))         # 104
print(percentile(latencies, 95))         # 120
print(percentile(latencies, 99))         # 900
```

Reading the output:

1. The average is 144.55 ms, which looks worse than almost every request because one value pulls it up.
2. The median, p50, is 104 ms, a better picture of a typical request.
3. p95 is 120 ms: 19 of the 20 requests finished in 120 ms or less.
4. p99 is 900 ms. With only 20 samples it is just the slowest one, which is why percentiles need many samples to be trustworthy.

## A common mistake {#mistake}
Timing a single run. One number tells you nothing about the spread, and the first run is slower anyway. Time many runs after a warm-up, and report percentiles together with the average.

## Key points {#key-points}
- Profile before optimizing, so you work on the real bottleneck.
- A bottleneck is the slowest step; speeding up anything else helps little.
- A benchmark keeps the conditions fixed so comparisons are fair.
- Discard warm-up runs; report p95 and p99 as well as the average.
