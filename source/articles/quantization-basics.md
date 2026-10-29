# Quantization: storing numbers with fewer bits

**Covers:** Quantization, Post-training quantization, PTQ, Quantization scale, Calibration data, Dequantize

## What problem does this solve? {#overview}
A model is a very long list of weights, and each weight takes space. Stored as FP32, every weight uses 4 bytes, so a model with 7 billion weights needs about 28 GB just to sit in memory. Many graphics cards cannot hold that, and moving that much data on every answer is slow.

**Quantization** fixes this by storing each number with fewer bits. An INT8 weight takes 1 byte instead of 4, so the same model shrinks to about 7 GB. The price is a small rounding error in every weight, which is why the rest of this article is about keeping that error under control.

## Post-training quantization {#post-training-quantization}
There are two ways to get a quantized model. You can train it with rounding in mind, or you can take a model that is already finished and convert it afterwards. The second way is **post-training quantization**, often written PTQ.

PTQ needs no new training run. You load the finished weights, choose how to round them, and save the smaller file. That makes it the usual first step: it takes minutes or hours instead of days, and it needs only a little data to guide the rounding.

## The quantization scale {#quantization-scale}
Integers are whole numbers, but weights are decimals such as 0.834. To bridge the two, quantization uses a conversion factor called the **quantization scale**.

For INT8 the integers run from −127 to 127. A common rule is to pick the scale so that the largest absolute weight lands exactly on 127:

scale = largest absolute weight ÷ 127

Each weight is divided by the scale and rounded to the nearest whole number. That whole number is what gets stored, along with one copy of the scale.

## Dequantize {#dequantize}
The stored integers are only useful if the math can turn them back into approximate real values. To **dequantize** is to multiply each integer by the scale. The result is close to the original weight but not identical, because rounding threw away the fine detail.

Some hardware can do arithmetic directly on integers and skip this step, which is one reason INT8 can be faster as well as smaller. Either way, the scale is what ties the integers back to real values.

## A worked example {#example}
Take four weights and quantize them to INT8.

```python
weights = [0.8, -0.3, 0.05, 1.27, 0.834]

scale = max(abs(w) for w in weights) / 127
stored = [round(w / scale) for w in weights]
restored = [q * scale for q in stored]

print(round(scale, 4))                      # 0.01
print(stored)                               # [80, -30, 5, 127, 83]
print([round(r, 3) for r in restored])      # [0.8, -0.3, 0.05, 1.27, 0.83]
```

The steps are:

1. The largest absolute weight is 1.27, so the scale is 1.27 ÷ 127 = 0.01.
2. Divide each weight by 0.01 and round: 0.8 becomes 80, and −0.3 becomes −30.
3. The last weight, 0.834, divides to 83.4, which rounds to 83.
4. Dequantize by multiplying by 0.01. The weight 0.834 returns as 0.83, an error of 0.004.

The first four weights came back exactly. Only the one that did not fall on a multiple of the scale lost detail.

## Calibration data {#calibration-data}
The scale above came from the weights themselves. For some parts of a model, such as the values flowing through it while it works, the range depends on what you feed in. **Calibration data** is a small sample of typical inputs, often a few hundred, that you run through the model only to measure those ranges. Nothing in the model changes during calibration; you only take measurements.

Poor calibration data gives a poor scale. If the sample looks nothing like real traffic, the scale will be too tight or too loose and accuracy will suffer.

## Why it can go wrong {#pitfalls}
Because the scale comes from the largest value, one huge weight stretches it. Add a weight of 12.7 to the example and the scale becomes 0.1, so 0.834 would be stored as 8 and come back as 0.8. The more the largest value exceeds the rest, the more detail everyone else loses. Later chapters show how to limit this damage.

## Key points {#key-points}
- Quantization stores numbers with fewer bits to save memory and often time.
- Post-training quantization converts a finished model without retraining it.
- The scale converts between integers and real values: real ≈ integer × scale.
- Dequantizing multiplies back by the scale; the result is close, not exact.
- Calibration data is a small sample used only to measure ranges, and it should resemble real inputs.
