# Continuous batching and serving engines

**Covers:** Static batching, Continuous batching, Serving engine, vLLM, TensorRT-LLM

## The waiting problem {#overview}
Grouping requests raises throughput, but language models make it awkward. Every request in a group writes a reply of a different length, and a reply is produced one token at a time. Some finish in five steps and some in fifty. The way you handle that difference decides how well the GPU is used, and it is the main reason purpose-built serving software exists.

## Static batching {#static-batching}
**Static batching** groups a fixed set of requests and runs them together until every one has finished. A request that ends early cannot leave: its slot sits idle, and the GPU keeps stepping the whole group until the longest reply is done. If one request in a group of four needs 50 steps and the others need 5, 8 and 12, three of the four slots are empty for most of the run. New requests also have to wait outside until the whole group is over.

## Continuous batching {#continuous-batching}
**Continuous batching** changes the group after every single token step. When a request finishes, its slot is freed at once and a waiting request takes it. The batch is always as full as the queue allows. No request is held back by a slower neighbour, and the GPU wastes far fewer steps.

## Serving engine {#serving-engine}
A **serving engine** is software that runs a model as a service. It receives requests, schedules them, forms batches, manages the memory for each conversation and returns answers. Continuous batching is not something you want to build yourself, so it comes built into serving engines. Using one means many users can share the same hardware at high throughput.

## vLLM {#vllm}
**vLLM** is an open-source serving engine for language models. It is known for continuous batching and for a memory technique called PagedAttention, which together let it serve many users at once on one GPU. It runs on several makes of hardware and is a common first choice.

## TensorRT-LLM {#tensorrt-llm}
**TensorRT-LLM** is NVIDIA's library for compiling and serving fast language-model engines. You get heavily tuned inference, often the fastest available on NVIDIA GPUs, but only on NVIDIA GPUs, and the models need a compile step first.

## A worked example {#example}
Eight requests with these reply lengths in steps, served by four slots. Compare the two schemes.

```python
def static_steps(lengths, slots):
    return sum(max(lengths[i:i + slots]) for i in range(0, len(lengths), slots))

def continuous_steps(lengths, slots):
    waiting, active, steps = list(lengths), [], 0
    while waiting or active:
        while waiting and len(active) < slots:
            active.append(waiting.pop(0))
        steps += 1
        active = [left - 1 for left in active if left - 1 > 0]
    return steps

lengths = [5, 50, 8, 12, 6, 9, 7, 10]
slots = 4
print(sum(lengths))                                  # 107 useful steps of work
print(static_steps(lengths, slots))                  # 60
print(continuous_steps(lengths, slots))              # 50
print(round(sum(lengths) / (slots * 60), 2))         # 0.45
print(round(sum(lengths) / (slots * 50), 2))         # 0.54
```

How to read it:

1. Together the requests need 107 token steps of real work.
2. Static batching runs two groups of four. The first lasts 50 steps because of the long reply, the second 10, so 60 steps in all.
3. Continuous batching refills slots as they free up, so the long reply overlaps with everything else and the total is 50 steps.
4. Slot use rises from 45% to 54%. The more uneven the lengths, the bigger the gain.

## A common mistake {#mistake}
Judging a serving engine by one benchmark with equal-length requests. Static batching looks fine when all replies have the same length. Test with the uneven lengths of real traffic.

## Key points {#key-points}
- Static batching holds every slot until the longest reply finishes.
- Continuous batching swaps finished requests out after every step, keeping the batch full.
- A serving engine bundles scheduling, batching and memory management.
- vLLM is an open-source engine; TensorRT-LLM is NVIDIA's compiled, NVIDIA-only option.
