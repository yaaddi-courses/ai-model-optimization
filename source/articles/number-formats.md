# Number formats: how a computer stores a weight

**Covers:** Bit, FP32, FP16, BF16, INT8, Overflow, Overflowing, Rounding error

## Why the format matters {#overview}
A weight is a number, and you might think storing a number is easy. But a computer cannot keep an unlimited number of digits. It has to pick a format, and the format decides three things at once: how much memory each number takes, how large or small a number can be, and how much fine detail survives. Changing the format is the most direct way to make a model smaller, so it is worth knowing the common ones. We will go through them one by one.

## Bit {#bit}
A **bit** is a single 0 or 1, the smallest unit of digital storage. Eight bits make a byte. A format with more bits can tell more values apart: with 8 bits you can represent 256 different patterns, and with 32 bits you can represent more than four billion.

The number of bits per weight is the dial that sets a model's size. A model with 7 billion weights at 32 bits (4 bytes) each needs 28 GB. At 16 bits it needs 14 GB, and at 8 bits it needs 7 GB.

## FP32 {#fp32}
**FP32** is 32-bit floating point, the standard full-precision format. A floating-point number is stored like scientific notation: a sign, an exponent that says how big the number is, and a fraction that holds the detail. FP32 spends 1 bit on the sign, 8 on the exponent and 23 on the fraction. It can reach about 3.4 × 10^38 and keeps roughly seven significant digits.

## FP16 {#fp16}
**FP16** is half precision, using 16 bits: 1 sign, 5 exponent and 10 fraction bits. That halves the memory of FP32. The cost is a small range. The largest number FP16 can hold is 65,504, so larger values cannot be stored.

## BF16 {#bf16}
**BF16**, or brain floating point, is also 16 bits, but it splits them differently: 1 sign, 8 exponent and 7 fraction bits. The 8 exponent bits are the same as FP32's, so BF16 covers the same huge range. It gives up detail instead of range, which makes it steadier than FP16 when values vary a lot.

## INT8 {#int8}
**INT8** is an 8-bit whole-number format that holds values from −128 to 127. It has no exponent at all, so it is compact and fast on hardware built for integers, at 1 byte per value. Turning decimals into INT8 needs a conversion factor, which a later article covers.

## Overflow {#overflow}
**Overflow** happens when a number is too large for the format to hold. The result may become infinity, wrap around or raise an error, and any of these can break the calculation. FP16 is more exposed than BF16 because its range stops at 65,504. When people say a value is overflowing, or that a calculation overflows, they mean it has passed that limit.

## Rounding error {#rounding-error}
**Rounding error** is the difference between a real number and the nearest value a format can store. Fewer bits means fewer storable values, so the gap gets larger. One such error is tiny, but a model combines billions of numbers, so the errors can add up and slightly hurt accuracy.

## A worked example {#example}
Python's `struct` module can store a number in FP16 and read it back, and that shows both limits at once.

```python
import struct

def through_fp16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]

print(through_fp16(0.1))        # 0.0999755859375
print(through_fp16(65504.0))    # 65504.0

try:
    through_fp16(70000.0)
except OverflowError as error:
    print("overflow:", error)   # overflow: float too large to pack with e format
```

What we see:

1. The number 0.1 comes back as 0.0999755859375. The difference, about 0.000024, is the rounding error.
2. 65,504 fits exactly, because it is the largest FP16 value.
3. 70,000 does not fit. FP16 cannot store it, and that is overflow.

In FP32, the same 0.1 comes back as 0.10000000149, much closer, and 70,000 fits easily.

## A common mistake {#mistake}
A common slip is assuming that fewer bits always means a worse model. Many models lose almost nothing at 16 or even 8 bits, because their weights do not need that much detail. The only way to know is to measure.

## Key points {#key-points}
- A bit is a 0 or 1, and the number of bits per weight sets memory use and detail.
- FP32 is the full-precision default, and FP16 halves memory but tops out at 65,504.
- BF16 keeps FP32's range with less detail, so it is steadier than FP16.
- INT8 stores whole numbers from −128 to 127 in one byte.
- Overflow breaks a calculation, and rounding error is the small loss every format has.
