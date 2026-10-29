# Computation graphs and graph compilers

**Covers:** Computation graph, Graph compiler, TensorRT, XLA, TVM, torch.compile, NumPy, PyTorch

## From code to a plan {#overview}
When you write a model in a framework, you write ordinary-looking code. The framework can also turn that code into a plan: a list of operations and how data flows between them. A tool that can see the whole plan can rearrange it to run faster, in ways that no one could do by looking at one operation at a time. That is the job of a graph compiler.

## Computation graph {#computation-graph}
A **computation graph** represents a model as operations, called nodes, connected by the tensors that flow between them, called edges. The expression `relu(a * b + c)` becomes a graph of three nodes: multiply, add, ReLU. The graph says what must be computed and in what order, but not how any single step is carried out. That separation is what allows optimization.

## Graph compiler {#graph-compiler}
A **graph compiler** analyses a model's computation graph and rewrites it into faster code for the target hardware. It can merge operations into fused kernels, remove steps that cancel out, reorder work to reuse memory, and choose the best implementation of each step for the chip. The model produces the same results, ideally, in less time.

## PyTorch and torch.compile {#torch-compile}
**PyTorch** is one of the most widely used open-source libraries for writing and training neural networks in Python. By default it runs code step by step as written. **torch.compile** is PyTorch 2's one-line way to hand a model's graph to a compiler: wrap the model, and the first call spends some time compiling, after which later calls run faster.

## TensorRT {#tensorrt}
**TensorRT** is NVIDIA's inference optimizer and compiler. It takes a trained model, fuses its operations, picks fast kernels and can use lower-precision numbers, producing an engine tuned for one NVIDIA GPU. It gives some of the best speed on NVIDIA hardware, and it runs only there.

## XLA {#xla}
**XLA** is Google's compiler for machine-learning graphs. It is used with JAX and TensorFlow and is the main route for running models on Google's TPUs, although it also targets GPUs and CPUs.

## TVM {#tvm}
Apache **TVM** is a free, community-built compiler that tunes models for a wide range of chips, from server GPUs to small phone and embedded processors. It is the choice when you need to reach hardware that the vendor tools do not cover.

## NumPy {#numpy}
**NumPy** is a Python library for fast numerical arrays. It is the basis of much scientific code, and PyTorch tensors resemble NumPy arrays. It is not a graph compiler: it runs each operation immediately and does not look at the whole plan, so it appears in questions as a name that does not belong in a list of compilers.

## A worked example {#example}
A tiny graph and one rewrite that a compiler would do, merging a multiply and an add.

```python
graph = [
    ("mul", "t1", ["a", "b"]),
    ("add", "t2", ["t1", "c"]),
    ("relu", "t3", ["t2"]),
]

def fuse_add_relu(nodes):
    fused = []
    i = 0
    while i < len(nodes):
        op, out, ins = nodes[i]
        if op == "add" and i + 1 < len(nodes) and nodes[i + 1][0] == "relu":
            fused.append(("add_relu", nodes[i + 1][1], ins))
            i += 2
        else:
            fused.append(nodes[i])
            i += 1
    return fused

print(len(graph), [n[0] for n in graph])                 # 3 ['mul', 'add', 'relu']
optimized = fuse_add_relu(graph)
print(len(optimized), [n[0] for n in optimized])         # 2 ['mul', 'add_relu']
```

How it works:

1. Each node is an operation, its output name and its inputs.
2. The compiler scans the list for an add directly followed by a ReLU.
3. It replaces the pair with one `add_relu` node that produces the ReLU's output.
4. The graph shrinks from three steps to two, and one temporary result disappears.

Real compilers apply hundreds of such rules, with checks that the result is still correct.

## A common mistake {#mistake}
Compiling a model once on a toy input and trusting the speed-up. Compilation often assumes the shapes it saw, so check the speed on realistic inputs and remember the first call is slow while it compiles.

## Key points {#key-points}
- A computation graph lists operations and the tensors between them.
- A graph compiler rewrites the graph into faster code for specific hardware.
- torch.compile, TensorRT, XLA and TVM are compilers from PyTorch, NVIDIA, Google and the open-source community.
- NumPy is an array library, not a compiler.
