# Model file formats: ONNX, GGUF and the Q levels

**Covers:** ONNX, GGUF, llama.cpp, JSON, Q8, Q4, Q3, Q2

## A model has to live in a file {#overview}
After training, a model is saved to a file so that other programs can load it. The format of that file decides which tools can open it, how big it is and how fast it loads. Two formats matter most for this course: ONNX, which moves models between tools, and GGUF, which packs quantized language models for running at home. Let's go through them and the names that come with them.

## ONNX {#onnx}
**ONNX**, the Open Neural Network Exchange, is an open file format that stores a model's computation graph together with its weights. Because it is a neutral standard, a model trained in one framework can be saved as ONNX and then run in another tool. It is the usual bridge between training code and optimized deployment tools.

## JSON {#json}
You may know **JSON** already: a plain-text data format for settings and small records, such as `{"name": "model", "layers": 12}`. It is easy to read, but it is not made for weights. Every number is written out as digits, so a weight that takes 2 bytes in a binary format takes around 15 bytes as text, and the file is slow to parse. Weights go in binary formats. JSON is for the small descriptions around them.

## GGUF {#gguf}
**GGUF** is a single-file model format used by llama.cpp and tools built on it, such as Ollama, for running quantized language models locally. One file holds the weights, the vocabulary and the settings, so there is nothing else to download or keep in step.

## llama.cpp {#llama-cpp}
**llama.cpp** is a free engine written in C and C++ that runs language models well on everyday computers, including ones with no big GPU. It reads GGUF files, runs well on CPUs and can use a GPU when there is one. That combination made local language models practical.

## The Q levels {#q-levels}
A GGUF file name often ends in a code like Q4. What does it mean? The number is roughly how many bits each weight uses after quantization, so a lower number means a smaller file and more quality loss.

- **Q8** is about 8 bits per weight, nearly as good as the original.
- **Q4** is about 4 bits, a popular balance of size and quality.
- **Q3** is about 3 bits, smaller again, with a noticeable loss.
- **Q2** is about 2 bits, the smallest file and the lowest quality.

The labels are approximate. The formats also store extra numbers for scales, so a real "Q4" file averages a little under 5 bits per weight.

## A worked example {#example}
Let's work out the file sizes for a model with 7 billion weights, using typical bits per weight, and then see how bulky text is for numbers.

```python
weights = 7_000_000_000
typical_bits = {"Q8": 8.5, "Q4": 4.85, "Q3": 3.91, "Q2": 2.63}

for name, bits in typical_bits.items():
    print(name, round(weights * bits / 8 / 1e9, 1), "GB")
# Q8 7.4 GB
# Q4 4.2 GB
# Q3 3.4 GB
# Q2 2.3 GB

as_text = "0.0999755859375"
print(len(as_text), "bytes as text versus 2 in a binary 16-bit format")
# 15 bytes as text versus 2 in a binary 16-bit format
```

What we learn:

1. Q8 is the largest, at 7.4 GB, close to the size of a plain 8-bit model.
2. Q4 saves almost half of that, at 4.2 GB.
3. Q3 and Q2 shrink the file further, to 3.4 and 2.3 GB, but each step costs quality.
4. The text version of a single weight takes 15 bytes against 2, which is why JSON is the wrong place for weights.

## A common mistake {#mistake}
A common slip is picking the lowest Q level because the file is smaller. Smaller is tempting, but the quality loss grows quickly. Try Q4 first, test on your own prompts, and move down only if the quality holds.

## Key points {#key-points}
- ONNX stores a graph plus weights and moves models between tools.
- GGUF is a single file for quantized language models, run by llama.cpp and tools like Ollama.
- The Q number is roughly the bits per weight, and lower means smaller and less accurate.
- JSON is fine for settings but wasteful for weights.
