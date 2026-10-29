# Inference runtimes: ONNX Runtime, OpenVINO and execution providers

**Covers:** Inference runtime, ONNX Runtime, OpenVINO, Execution provider

## Running a saved model {#overview}
A file full of weights does nothing on its own, much like a recipe does not cook anything. Something must load it, send it inputs and run the arithmetic on the chip that is available. You could write all of that by hand for every device, but nobody wants to. A runtime does it for you, and in this article we see how.

## Inference runtime {#inference-runtime}
An **inference runtime** is the program that loads a trained model and carries out its arithmetic efficiently on the chip you have. You can ask for predictions without writing any low-level code. The runtime handles memory, picks fast implementations of each operation, and often applies optimizations such as fusing steps. Runtimes only run models. They do not train them.

## ONNX Runtime {#onnx-runtime}
**ONNX Runtime** is the engine that executes ONNX models quickly on a wide range of chips. You give it the saved model file, and it takes care of the rest. Because ONNX is a neutral format, the same model file can run on a server GPU, a plain CPU or a phone, using the same code.

## Execution provider {#execution-provider}
How does one runtime talk to so many kinds of hardware? Inside ONNX Runtime, an **execution provider** is a pluggable back end for one kind of hardware. There is a provider for CPUs, one for NVIDIA GPUs through CUDA, one for TensorRT, and others. When the runtime loads a model, you give it a list of providers in order of preference. It hands each part of the model to the first provider that can run it, and falls back to the next one for the rest.

## OpenVINO {#openvino}
**OpenVINO** is Intel's toolkit for running trained models quickly on Intel hardware. It speeds up inference on CPUs, integrated GPUs and NPUs without you rewriting the model. It can also work as an execution provider inside ONNX Runtime, so you can reach Intel chips with the same model file.

## A worked example {#example}
Let's see how an ordered list of providers behaves on three different machines.

```python
def choose_provider(available, preferred):
    for name in preferred:
        if name in available:
            return name
    raise RuntimeError("no usable provider")

preferred = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]

gpu_server = ["CUDAExecutionProvider", "CPUExecutionProvider"]
intel_laptop = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
plain_box = ["CPUExecutionProvider"]

print(choose_provider(gpu_server, preferred))     # CUDAExecutionProvider
print(choose_provider(intel_laptop, preferred))   # OpenVINOExecutionProvider
print(choose_provider(plain_box, preferred))      # CPUExecutionProvider
```

What happens on each machine:

1. The preferred list puts the fastest option first and the CPU last, as a safe default.
2. On the GPU server, the first choice is available and gets used.
3. The Intel laptop has no CUDA, so it falls through to OpenVINO.
4. The plain machine falls through to the CPU. The same model file and the same list work on all three.

The CPU entry at the end means the program always runs somewhere, even if slowly.

## A common mistake {#mistake}
A common slip is assuming the fastest provider is used automatically. If the package for that provider is not installed, the runtime quietly falls back to the CPU. Everything works, only slowly, so nothing warns you. Check which provider the session actually reports.

## Key points {#key-points}
- A runtime loads a trained model and runs it efficiently on the hardware you have.
- ONNX Runtime runs ONNX models on many devices.
- An execution provider is a back end for one kind of hardware, tried in your order of preference.
- OpenVINO targets Intel chips and can also plug into ONNX Runtime.
