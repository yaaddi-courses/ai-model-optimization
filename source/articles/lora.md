# LoRA, adapters and QLoRA: cheap fine-tuning

**Covers:** LoRA, Adapter, QLoRA

## The cost of customizing {#overview}
Suppose you want a large model to follow your company's writing style. Fine-tuning it the usual way updates every weight. That needs memory for the weights, for the gradients and for extra training state, which together can be several times the size of the model itself. It also produces a full copy of the model for each task, so you end up storing and serving a lot of large files. A group of methods avoids both problems: they train only a small add-on and leave the original untouched.

## Adapter {#adapter}
An **adapter** is a small set of extra trainable weights attached to a frozen base model. The base weights never change; only the adapter learns your task. To use the result, you load the base model once and plug in the adapter. A single large model can then serve many tasks by swapping adapters, which are small files, instead of loading a separate full model for each.

## LoRA {#lora}
How is an adapter built in practice? The most popular way is **LoRA**, short for low-rank adaptation. It freezes a weight matrix and learns its change as the product of two thin matrices. This is the idea of low-rank factorization, applied to the update instead of to the weights. If the original matrix is 4,096 by 4,096, the update is built from a 4,096 by 8 matrix and an 8 by 4,096 matrix. The number 8 is the rank, and it sets how much the adapter can learn.

After training, you can keep the two thin matrices as a separate adapter file. Or you can add their product into the original weights, so the model runs at its original speed with no extra step.

## QLoRA {#qlora}
**QLoRA** goes one step further and combines LoRA with compression of the base model. The frozen base is stored in 4-bit form, which saves a great deal of memory, and the small LoRA matrices are trained on top at higher precision. The base weights are converted back to a usable form only when a calculation needs them. The result is that fairly large models can be fine-tuned on a single modest GPU.

## A worked example {#example}
Let's measure how small a LoRA update is for one 4,096 by 4,096 matrix at rank 8.

```python
size = 4096
rank = 8

full_update = size * size
lora_update = 2 * size * rank

print(full_update)                              # 16777216
print(lora_update)                              # 65536
print(round(100 * lora_update / full_update, 2))    # 0.39

layers = 32
matrices_per_layer = 4
trainable = lora_update * matrices_per_layer * layers
print(trainable)                                # 8388608
print(round(trainable * 2 / 1e6, 1))            # 16.8 MB at 2 bytes per number
```

Step by step:

1. Updating the whole matrix would mean training 16.8 million numbers.
2. LoRA trains two thin matrices with 65,536 numbers in total, which is 0.39% as many.
3. Apply it to four matrices in each of 32 layers, and the adapter has about 8.4 million numbers.
4. At 2 bytes each, the adapter file is around 17 MB, tiny next to a model of many gigabytes.

That is why adapters are easy to store, share and swap.

## A common mistake {#mistake}
A common slip is choosing a rank that is too small and then blaming the method. If the adapter cannot reach the quality you need, raise the rank and compare. A low rank is cheap, but it has limited room to learn.

## Key points {#key-points}
- An adapter is a small trainable add-on attached to a frozen base model.
- LoRA learns the weight change as a product of two thin matrices.
- Adapters are small files, so one base model can serve many tasks.
- QLoRA stores the base in 4-bit form so large models can be tuned on a modest GPU.
