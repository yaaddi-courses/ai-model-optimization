# FP8 and its variants: eight bits that still float

**Covers:** FP8, E4M3, E5M2, E8M7, E2M5

## What is FP8? {#overview}
You may already know that floating-point numbers such as FP16 and BF16 store a sign, an exponent and a fraction. **FP8** applies the same idea in only 8 bits, which is half the memory of FP16. Newer GPUs have hardware that works directly on FP8, so it can make a model both smaller and faster.

There is a catch. Eight bits leave very little room. Once the sign takes one bit, the remaining seven must be shared between the exponent and the fraction. Every split is a compromise, and that is why there is more than one FP8 format.

## Reading the names {#naming}
The variants are named by how they spend their bits. The letter E stands for exponent bits, and M stands for mantissa bits, the fraction that holds the detail. So E4M3 means 4 exponent bits and 3 fraction bits. The sign bit is not counted in the name. Add them up and you get 1 + 4 + 3 = 8.

Why does the split matter? Exponent bits buy range, meaning how large and how small a number can be. Fraction bits buy precision, meaning how finely two nearby numbers can be told apart. You cannot have both from the same bits.

## E4M3 {#e4m3}
**E4M3** has 4 exponent bits and 3 fraction bits, and it favours precision. Its largest value is 448, so its range is narrow, but within that range its numbers sit closer together. It is the usual choice for weights and for values that stay in a modest range.

## E5M2 {#e5m2}
**E5M2** has 5 exponent bits and 2 fraction bits, and it favours range. Its largest value is 57,344, but with only 2 fraction bits its numbers are spaced further apart. It suits values that can swing widely, such as the gradients used in training.

## E8M7 {#e8m7}
**E8M7** is sometimes mentioned next to the FP8 formats, but it is not one. Add the bits: 1 + 8 + 7 = 16. That is BF16's split of 8 exponent bits and 7 fraction bits. If a question offers E8M7 as an FP8 variant, it is a distractor.

## E2M5 {#e2m5}
**E2M5** is a hypothetical 8-bit split with 2 exponent bits and 5 fraction bits. Hardware does not support it as a standard format. The real FP8 formats are E4M3 and E5M2. A name that sounds right is not proof that a format exists, so check which formats your GPU supports before you plan around one.

## A worked example {#example}
We can compare the range and the spacing of the two real formats with a few lines of arithmetic.

```python
formats = {
    "E4M3": {"exp_bits": 4, "frac_bits": 3, "largest": 448},
    "E5M2": {"exp_bits": 5, "frac_bits": 2, "largest": 57344},
}

for name, f in formats.items():
    steps_per_doubling = 2 ** f["frac_bits"]
    print(name, "largest:", f["largest"], "| steps per doubling:", steps_per_doubling)
# E4M3 largest: 448 | steps per doubling: 8
# E5M2 largest: 57344 | steps per doubling: 4

print(57344 / 448)   # 128.0
```

Reading the output:

1. Between any value and its double, E4M3 can represent 8 distinct numbers. E5M2 can represent only 4.
2. So E4M3 is twice as fine-grained near any given value.
3. E5M2 reaches 57,344, which is 128 times the largest E4M3 value.
4. Neither is better in general. The right one depends on whether your numbers need precision or reach.

## A common mistake {#mistake}
A common slip is switching a whole model to one FP8 format without checking the values. If a value passes the format's maximum, it overflows. A model with wide-ranging values may need E5M2 or a different scale.

## Key points {#key-points}
- FP8 is an 8-bit floating-point format with half the memory of FP16.
- In names like E4M3, E counts exponent bits (range) and M counts fraction bits (precision).
- E4M3 is more precise and reaches 448, and E5M2 reaches 57,344 with less precision.
- E8M7 is a BF16-style 16-bit split and E2M5 is hypothetical, so neither is a real FP8 variant.
