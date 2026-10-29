# LoRA, adapters and QLoRA: cheap fine-tuning

**Covers:** LoRA, Adapter, QLoRA

## The cost of customizing {#overview}
Fine-tuning a large model normally updates every weight. That needs memory for the weights, for the gradients and for extra training state, which can be several times the size of the model itself. It also produces a full copy of the model for each task, which is a lot of files to store and serve. A group of methods gets around both problems by training only a small add-on and leaving the original untouched.

## Adapter {#adapter}
An **adapter** is a small set of extra trainable weights attached to a frozen base model. The base model's weights are never changed. Only the adapter learns your task. To use the result you load the base model once and plug in the adapter. A single large model can then serve many tasks by swapping adapters, which are small files, instead of loading a separate full model for each.

## LoRA {#lora}
**LoRA**, short for low-rank adaptation, is the most popular adapter method. It freezes a weight matrix and learns its change as the product of two thin matrices, the same idea as low-rank factorization applied to the update instead of to the weights. If the original matrix is 4,096 by 4,096, the update is built from a 4,096 by 8 matrix and an 8 by 4,096 matrix. The number 8 is the rank, and it sets how much the adapter can learn.

After training, you can keep the two thin matrices as a separate adapter file, or add their product into the original weights so the model runs at its original speed with no extra step.

## QLoRA {#qlora}
**QLoRA** combines LoRA with compression of the base model. The frozen base is stored in 4-bit form, which saves a great deal of memory, and the small LoRA matrices are trained on top at higher precision. The base weights are converted back to a usable form only when needed for a calculation. The result is that fairly large models can be fine-tuned on a single modest GPU.

## A worked example {#example}
How small a LoRA update is, for one 4,096 by 4,096 matrix, at rank 8.

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

How to read it:

1. Updating the whole matrix would mean training 16.8 million numbers.
2. LoRA trains two thin matrices with 65,536 numbers in total, 0.39% as many.
3. Apply it to four matrices in each of 32 layers and the adapter has about 8.4 million numbers.
4. At 2 bytes each, the adapter file is around 17 MB, tiny next to a model of many gigabytes.

That is why adapters are easy to store, share and swap.

## A common mistake {#mistake}
Using a rank that is too small and blaming the method. If the adapter cannot reach the quality you need, raise the rank and compare. A low rank is cheap but has limited room to learn.

## Key points {#key-points}
- An adapter is a small trainable add-on attached to a frozen base model.
- LoRA learns the weight change as a product of two thin matrices.
- Adapters are small files, so one base model can serve many tasks.
- QLoRA stores the base in 4-bit form so large models can be tuned on a modest GPU.
