# Choosing a strategy, and the pitfalls to avoid

**Covers:** Low-hanging fruit, Accuracy floor, Premature optimization, Latency-throughput trade-off, UI

## Where to start {#overview}
By now you know many techniques: shrinking numbers, removing weights, batching, caching, compiling, routing. Using all of them is neither possible nor wise, because each one costs effort, adds risk and brings complexity. The skill is in choosing which to apply, in what order, and when to stop. In this article we build a simple order of attack, and we look at the traps that catch people along the way.

## Low-hanging fruit {#low-hanging-fruit}
Where should you begin? With the **low-hanging fruit**: optimizations that are cheap and low-risk to apply. Examples are caching repeated answers, batching requests and choosing a standard lower-precision format. They need little work, you can undo them, and they often deliver the biggest share of the gain. Try them first, before harder changes such as retraining or building something custom.

## Accuracy floor {#accuracy-floor}
Before you start, decide where you will stop. The **accuracy floor** is the lowest quality level you will accept. Any optimization that drops results below it is rejected, however much faster or cheaper it is. It is the other side of the accuracy budget: the budget says how much you may lose, and the floor says where you must stop. Set it before you begin, together with the people who own the product.

## Premature optimization {#premature-optimization}
**Premature optimization** is spending effort on speed before you have measured whether it is the real bottleneck. It wastes time and adds complexity for no gain, and complicated code is harder to maintain. The habit that prevents it is simple: measure first, fix the slowest step, then measure again. A change that is not backed by a number is a guess.

## Latency-throughput trade-off {#latency-throughput-trade-off}
The **latency-throughput trade-off** is the tension where a bigger batch size serves more requests per second overall, but makes each individual request wait longer. You cannot maximise both. Pick the point that fits your product: a live chat wants low latency, and a nightly batch job wants high throughput.

## UI {#ui}
**UI** stands for user interface, the screens and buttons through which people use software. It matters here because users do not feel your metrics. They feel the interface. Showing text as it is produced makes a five-second reply feel quick, and a progress indicator makes a delay bearable. Sometimes the cheapest speed-up is in the UI, not in the model.

## A worked example {#example}
Let's compute throughput for three batch sizes, and then rank some options by gain per unit of effort.

```python
runs = [(1, 100), (8, 160), (32, 400)]      # (batch size, milliseconds per batch)

for batch, ms in runs:
    print(batch, ms, round(batch / (ms / 1000), 1))
# 1 100 10.0
# 8 160 50.0
# 32 400 80.0

options = [
    ("cache repeated answers", 30, 2),
    ("lower-precision weights", 40, 3),
    ("custom kernels", 15, 20),
]
for name, gain, effort in sorted(options, key=lambda o: o[1] / o[2], reverse=True):
    print(name, round(gain / effort, 1))
# cache repeated answers 15.0
# lower-precision weights 13.3
# custom kernels 0.8
```

Reading the output:

1. A batch of one answers in 100 ms and handles 10 requests per second.
2. Batches of 8 handle 50 per second, but each request now waits 160 ms.
3. Batches of 32 reach 80 per second, with a 400 ms wait. Throughput grew 8 times while latency grew 4 times.
4. The options list gives each guess a percent gain and days of effort. Caching and lower precision come first, while custom kernels return little for the work.

The numbers are made up to show the method. In your own project, use measured ones.

## A common mistake {#mistake}
A common slip is optimizing for the best number on a single metric. A model that is 3 times faster but breaks the accuracy floor, or a server with record throughput and unhappy users, has solved the wrong problem. Keep the goal in view: users getting good answers at an acceptable cost.

## Key points {#key-points}
- Start with the cheap, safe changes before the hard ones.
- Set an accuracy floor first, and reject anything that crosses it.
- Do not optimize before measuring what is really slow.
- Bigger batches raise throughput and latency together, so choose by product need.
- Improving what the user sees in the UI can be the cheapest speed-up.
