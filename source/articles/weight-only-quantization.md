# Weight-only quantization: GPTQ, AWQ and groups

**Covers:** Weight-only quantization, GPTQ, AWQ, Group-wise quantization, Group size, Outlier

## Shrinking only the weights {#overview}
Where does a language model keep most of its memory? In its weights. The values that flow through the model while it works, the activations, are comparatively small, and they change with every input. This suggests a practical shortcut: shrink only the weights, and leave the activations at higher precision. The delicate part of the calculation stays accurate, and most of the memory is still saved.

In this article we look at that idea and the tools built on it. They matter because 4-bit weight-only methods are how large models are squeezed onto ordinary GPUs.

## Weight-only quantization {#weight-only-quantization}
**Weight-only quantization** stores the weights in low precision, such as 4 or 8 bits, while the activations stay at 16 bits. When the model runs, the stored weights are converted back to approximate real numbers at the moment they are needed, and then multiplied with the activations.

The main benefit is memory and file size. A modest speed-up often follows, because less data has to be read from memory for each answer. The weights must still be converted back, so the arithmetic itself is not faster than before.

## Outlier {#outlier}
Here is the part that trips people up. An **outlier** is a weight whose value is unusually large compared with its neighbours. Quantization picks its conversion factor from the biggest value in the set, so one outlier stretches the factor for everyone. All the ordinary weights then land on only a few of the available integer steps and lose their detail. Large language models are known for having a few such weights, which is why plain rounding often works poorly on them. The word is used for any extreme value, not only weights. A handful of very slow requests among thousands are outliers too, and for the same reason an average hides them.

## Group-wise quantization {#group-wise-quantization}
**Group-wise quantization** limits the damage. Instead of one factor for a whole matrix, the weights are split into small groups, and each group gets its own factor. An outlier then harms only the neighbours in its own group.

## Group size {#group-size}
The **group size** is the number of weights that share one factor. Typical values run from 32 to 128. A smaller group follows the data more closely and loses less, but it needs more factors stored. A larger group is cheaper to store and less accurate. At a group size of 128 with 16-bit factors, the extra storage is only about 0.125 bits per weight.

## GPTQ {#gptq}
**GPTQ** is a post-training method for 4-bit weight-only quantization. It works through the model layer by layer. Within a layer, it rounds the weights one at a time and, after each, adjusts the remaining weights to cancel the error it made. It uses a small sample of real inputs to know which errors matter, and no full retraining is needed.

## AWQ {#awq}
**AWQ**, short for activation-aware weight quantization, starts from a different observation: a few weights matter far more than the rest, namely those that multiply large activations. It looks at activations on calibration data, finds those important weights, and scales them up before rounding, so that rounding hurts them less. The result is usually a small accuracy loss at 4 bits.

## A worked example {#example}
Let's take eight weights, one of them an outlier. We quantize them to INT8, first with a single factor and then in two groups of four.

```python
weights = [0.12, -0.25, 0.31, 0.08, 0.15, -0.2, 6.35, 0.05]

def round_trip(values):
    scale = max(abs(v) for v in values) / 127
    return [round(v / scale) * scale for v in values]

single = round_trip(weights)
grouped = round_trip(weights[:4]) + round_trip(weights[4:])

def worst_error(original, restored):
    return max(abs(a - b) for a, b in zip(original, restored))

print(round(worst_error(weights[:4], single[:4]), 4))    # 0.02
print(round(worst_error(weights[:4], grouped[:4]), 5))   # 0.00102
```

What happens:

1. With one factor, the outlier 6.35 sets it to 0.05, so 0.12 is stored as 0.10 and 0.08 as 0.10.
2. The first four weights are wrong by up to 0.02, a large error for values around 0.1.
3. With two groups, the first group has its own factor, about 0.0024.
4. The first four weights now come back with an error of only about 0.001, twenty times smaller.

The outlier's group is still coarse, but the damage no longer spreads.

## A common mistake {#mistake}
A common slip is choosing the smallest group size without counting the cost. Each group stores a factor, so very small groups can eat much of the memory you saved. Measure accuracy and size together.

## Key points {#key-points}
- Weight-only quantization shrinks the weights and keeps activations at higher precision.
- An outlier stretches the conversion factor and costs its neighbours detail.
- Group-wise quantization gives each small group its own factor, and group size is how many weights share one.
- GPTQ rounds weights one by one and corrects the rest, and AWQ protects the weights that meet large activations.
