# Optimizing AI Models for Inference — deck list

Approved 2026-09-21 (29 decks, foundations -> practice -> shipping).
Audience: developers new to ML. Tap-only, concept-first.

## Foundations
1. What a Model Is (Weights, Parameters, Tensors)
2. Training vs Inference
3. Latency, Memory and Money
4. Measure First: Profiling and Benchmarks
5. Accuracy vs Speed Trade-offs

## Making Models Smaller
6. Numeric Precision (FP32, FP16, BF16, INT8)
7. Post-Training Quantization
8. Quantization-Aware Training
9. Pruning
10. Knowledge Distillation
11. Low-Rank and Weight Sharing

## Faster Compute
12. CPU, GPU and Accelerators
13. Batching
14. Operator Fusion and Graph Compilers
15. Model Formats (ONNX, GGUF)
16. Runtimes (TensorRT, ONNX Runtime, llama.cpp)

## LLM-Specific
17. Tokens and Context Length
18. The KV Cache
19. Attention Optimizations (FlashAttention)
20. Speculative Decoding
21. Continuous Batching and Serving Engines
22. LoRA and QLoRA

## Serving and Cost
23. Caching and Routing
24. Autoscaling and Cold Starts
25. Cost per Request
26. Edge and On-Device Deployment

## Validating and Shipping
27. Regression-Testing After Optimization
28. Choosing a Strategy
29. Common Pitfalls
