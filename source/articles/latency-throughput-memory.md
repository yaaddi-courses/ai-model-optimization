# Latency, throughput and memory: the three numbers that matter

**Covers:** Latency, Throughput, Memory footprint, Activations, GB, RAM

## Why three numbers {#overview}
When people say a model is "slow" or "expensive", they usually mean one of three different things. It may answer too slowly for one user. It may serve too few users at once. Or it may not fit on the machine at all. Each problem has its own measure and its own fixes, so it helps to keep them apart from the start. Let's meet the three numbers.

## Latency {#latency}
You have felt latency every time you waited for a page to load. **Latency** is the time one request takes, from being sent to getting its answer. If you ask a model a question and wait 200 milliseconds, the latency is 200 ms.

It is what a single person feels. For a chat box or a voice assistant, lower is better. For a nightly report that nobody is watching, latency hardly matters.

## Throughput {#throughput}
**Throughput** is how much work the system finishes per unit of time, such as requests per second. It describes the whole service, not one person. Higher throughput means the same hardware serves more users, which lowers the cost of each answer.

Latency and throughput are related, but they are not the same thing. A system can have low latency and low throughput, such as a quiet server that answers one person very fast. It can also have high throughput and high latency, such as a busy server that finishes thousands of requests per second while each one waits in line.

## Memory footprint {#memory-footprint}
The **memory footprint** is how much memory a model needs while it runs. The biggest part is the weights, and the rest is working space.

Memory decides whether a model can run at all. A GPU with 24 GB of its own memory cannot hold a model that needs 28 GB, however fast the chip is. A GB, or gigabyte, is about one billion bytes, and it is the usual unit for model sizes. RAM is the main working memory of an ordinary computer, where running programs keep their data. A GPU has its own separate memory, and the weights usually need to fit there for fast answers.

## Activations {#activations}
**Activations** are the intermediate results each layer computes while it processes an input. They are not part of the saved model, but they take space while it runs, next to the weights. Longer inputs and larger groups of requests make them bigger. This is why the memory you need on a busy day can be more than the size of the model file.

## A worked example {#example}
Let's try one. Suppose a model has 7 billion weights, and each weight takes 2 bytes. A server answers one request at a time in 200 ms, or four requests together in 250 ms.

```python
weights = 7_000_000_000
bytes_per_weight = 2
print(weights * bytes_per_weight / 1e9)   # 14.0 GB for the weights alone

one_at_a_time_latency = 0.200
print(1 / one_at_a_time_latency)          # 5.0 requests per second

together_latency = 0.250
print(4 / together_latency)               # 16.0 requests per second
```

Reading the results:

1. Memory: 7 billion × 2 bytes is 14 GB, before any activations are counted.
2. One at a time, the server finishes a request every 0.2 seconds, so throughput is 5 requests per second.
3. Doing four together takes 0.25 seconds each time, so throughput rises to 16 requests per second.
4. The price is latency. Each user now waits 250 ms instead of 200 ms.

Higher throughput for slightly higher latency is a typical trade, and later articles return to it.

## A common mistake {#mistake}
A common slip is to report only one number. The average latency alone hides the experience of unlucky users, and throughput alone hides how long people wait. Quote both, and always say how much memory the setup needs.

## Key points {#key-points}
- Latency is how long one request takes, and throughput is how many requests finish per second.
- Memory footprint decides whether a model fits on the hardware at all.
- Activations are working results that take memory on top of the weights.
- Improving one of the three can hurt another, so measure all of them.
