# Quantization-aware training: learning to live with rounding

**Covers:** Quantization-aware training, QAT, Fake quantization

## When rounding after the fact is not enough {#overview}
Post-training quantization converts a finished model, and it often works well at 8 bits. At lower precision the rounding errors grow, and a model that was never prepared for them can lose too much accuracy. If you cannot accept that loss, the next option is to prepare the model for the rounding while it is still learning. That is the idea behind quantization-aware training.

## Quantization-aware training {#quantization-aware-training}
**Quantization-aware training**, written QAT, trains or fine-tunes a model while simulating low-precision rounding. During each forward pass, the weights are rounded to the values the final low-bit format can hold. The model therefore sees the effect of the errors, and the training process adjusts the weights so the model works well despite them.

You usually start from a pretrained model and run a short fine-tuning with the simulation switched on, rather than training from scratch. After QAT, converting to real low-bit numbers loses little, because the model already learned to cope.

## Fake quantization {#fake-quantization}
The simulation has a name: **fake quantization**. It rounds each value to the nearest level that the low-precision format could store, but then keeps the result in ordinary floating point. The model feels the error, yet the numbers are still in a format that training can handle.

This matters because training needs gradients, and rounding to integers breaks them: the rounding step is flat almost everywhere, with sudden jumps. Training libraries get around that by treating the rounding as if it passed the value straight through when they compute gradients. You will see this called a straight-through estimator, though you do not need the name to use the idea.

## A worked example {#example}
A weight is 0.37. The low-precision grid in this toy case has steps of 0.1, so the weight can only be stored as 0.4 or 0.3.

```python
def fake_quantize(value, step=0.1):
    return round(round(value / step) * step, 2)

weight = 0.37
print(fake_quantize(weight))        # 0.4

# the model computes with 0.4, sees a slightly wrong answer,
# and training nudges the real weight toward a value that works on the grid
weight = 0.34
print(fake_quantize(weight))        # 0.3
```

Following it:

1. The weight 0.37 is rounded to 0.4 for the forward pass, so the output uses 0.4.
2. If 0.4 gives a worse result than 0.3 would have, the gradient says the weight should shrink.
3. After an update, the weight is 0.34, which now rounds to 0.3.
4. Over many steps the weights settle on values that sit well on the grid and give good answers.

A post-training method cannot do this, because by then the weights no longer change.

## When to use it {#when-to-use}
QAT costs a training run, so use it when the cheap method fails your accuracy budget, or when you aim for very low precision. If post-training quantization already meets the budget, QAT adds expense for no benefit. Compared with pruning, which removes weights, QAT keeps every weight and changes how it is stored.

## A common mistake {#mistake}
Believing that fake quantization shrinks the model. It does not. During QAT the numbers stay in floating point, and the model file is no smaller. The size drops only when you export to the real low-bit format afterwards.

## Key points {#key-points}
- QAT trains or fine-tunes while simulating low-precision rounding.
- Fake quantization rounds values to the low-bit grid but keeps them in floating point.
- The model learns weights that work well after real conversion.
- It costs training time, so try post-training quantization first.
