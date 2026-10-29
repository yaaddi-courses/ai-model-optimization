# Continuous batching and serving engines

**Covers:** Static batching, Continuous batching, Serving engine, vLLM, TensorRT-LLM

## The waiting problem {#overview}
You know that grouping requests raises throughput. Language models make this awkward, though. Every request in a group writes a reply of a different length, and a reply is produced one token at a time. Some finish in five steps and some need fifty. How you handle that difference decides how well the GPU is used, and it is the main reason purpose-built serving software exists.

## Static batching {#static-batching}
The simplest approach is **static batching**: take a fixed set of requests and run them together until every one has finished. Picture a bus that cannot let anyone off until the last stop. A request that ends early cannot leave, so its slot sits idle while the GPU keeps stepping the whole group until the longest reply is done. If one request in a group of four needs 50 steps and the others need 5, 8 and 12, three of the four slots are empty for most of the run. New requests also wait outside until the whole group is over.

## Continuous batching {#continuous-batching}
**Continuous batching** removes that waiting. It changes the group after every single token step. When a request finishes, its slot is freed at once and a waiting request takes it. The batch stays as full as the queue allows. No request is held back by a slower neighbour, and the GPU wastes far fewer steps.

## Serving engine {#serving-engine}
Nobody wants to build this scheduling by hand, and that is where a **serving engine** comes in. It is the software that runs a model as a service: it receives requests, schedules them, forms batches, manages the memory for each conversation and returns the answers. Continuous batching comes built into serving engines. With one, many users can share the same hardware at high throughput.

## vLLM {#vllm}
**vLLM** is an open-source serving engine for language models. It is known for continuous batching and for a memory technique called PagedAttention, which together let it serve many users at once on one GPU. It runs on several makes of hardware, and many teams choose it first.

## TensorRT-LLM {#tensorrt-llm}
**TensorRT-LLM** is NVIDIA's library for compiling and serving fast language-model engines. You get heavily tuned inference, often the fastest available on NVIDIA GPUs. The catch is that it works only on NVIDIA GPUs, and models need a compile step first.

## A worked example {#example}
Let's compare the two schemes on eight requests, served by four slots. The lengths below are the reply lengths in steps.

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

Here is what happens:

1. Together the requests need 107 token steps of real work.
2. Static batching runs two groups of four. The first lasts 50 steps because of the long reply, the second lasts 10, so the total is 60 steps.
3. Continuous batching refills slots as they free up. The long reply overlaps with everything else, and the total is 50 steps.
4. Slot use rises from 45% to 54%. The more uneven the lengths, the bigger the gain.

## A common mistake {#mistake}
A common slip is judging a serving engine with one benchmark in which every request has the same length. Static batching looks fine when all replies are equal, so the test hides the very problem continuous batching solves. Test with the uneven lengths of real traffic.

## Key points {#key-points}
- Static batching holds every slot until the longest reply finishes.
- Continuous batching swaps finished requests out after every step, which keeps the batch full.
- A serving engine bundles scheduling, batching and memory management.
- vLLM is an open-source engine, and TensorRT-LLM is NVIDIA's compiled, NVIDIA-only option.
