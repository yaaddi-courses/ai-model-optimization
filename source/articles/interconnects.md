# Interconnects: NVLink and PCIe

**Covers:** Interconnect, NVLink, PCIe

## The wires between the chips {#overview}
When a model is split across several GPUs, the GPUs must constantly pass results to each other. The speed of that exchange can matter as much as the speed of the chips. A cluster of the fastest GPUs joined by slow links spends much of its time waiting, like a team of quick workers who share one narrow doorway. The hardware that carries the data is called the interconnect, and in this article we compare the two you will meet most.

## Interconnect {#interconnect}
An **interconnect** is the wiring that moves data between GPUs, or between GPUs and the rest of the machine. Its bandwidth limits how well a model can be split. Tensor parallelism needs the GPUs to swap data after every layer, so a slow interconnect can turn a speed-up into a slowdown.

## PCIe {#pcie}
**PCIe** is the general-purpose expansion slot connection in a computer. Graphics cards, storage and network cards all plug into it. It is everywhere, so GPUs always have it. For GPU-to-GPU traffic it is the slower route, because the data goes through the connection to the computer's central parts. Each new version doubles the speed of the last: roughly 32 GB per second in each direction for version 4, and 64 for version 5, for a full-size slot.

## NVLink {#nvlink}
**NVLink** is NVIDIA's high-bandwidth link that connects GPUs directly to one another. It is typically many times faster than PCIe, at several hundred gigabytes per second per GPU. It exists only on NVIDIA's data-centre hardware, and it needs a machine designed to use it. When people say a server has "eight GPUs on NVLink", they mean the GPUs can exchange data at full speed.

## How machines are wired {#wiring}
Inside one server, GPUs are often joined by NVLink through a dedicated switch, so any GPU can reach any other at full speed. Between servers, the link is much slower: ordinary network cards move a few tens of gigabytes per second at best. This gives you a rule of thumb for splitting a model. Keep the chatty parts, such as tensor parallelism, inside one server where the fast links are, and use the lighter pipeline or data approaches across servers.

When you read a spec sheet, look for two numbers. The first is the bandwidth of the GPU-to-GPU link. The second is how many GPUs share that link at full speed. A server that advertises eight GPUs but links them in two groups of four has a fast path only within each group.

## A worked example {#example}
Let's see how long it takes to move one gigabyte over each link, using the round speeds above, and then what that means for a split model.

```python
speeds_gb_per_s = {"NVLink": 900, "PCIe 5": 64, "PCIe 4": 32}

for name, speed in speeds_gb_per_s.items():
    print(name, round(1 / speed * 1000, 2), "ms per GB")
# NVLink 1.11 ms per GB
# PCIe 5 15.62 ms per GB
# PCIe 4 31.25 ms per GB

layers = 80
gb_exchanged_per_layer = 0.1
for name, speed in speeds_gb_per_s.items():
    total_ms = layers * gb_exchanged_per_layer / speed * 1000
    print(name, round(total_ms, 1), "ms of waiting per request")
# NVLink 8.9 ms of waiting per request
# PCIe 5 125.0 ms of waiting per request
# PCIe 4 250.0 ms of waiting per request
```

Reading the output:

1. One gigabyte takes about 1.1 ms over NVLink, 15.6 ms over PCIe 5 and 31.3 ms over PCIe 4.
2. Suppose a split model exchanges 0.1 GB after each of its 80 layers, so 8 GB in all.
3. Over NVLink that adds about 9 ms to a request. Over PCIe 4 it adds 250 ms.
4. A quarter of a second of waiting would erase much of the benefit of using several GPUs.

The figures are round, and real speeds vary by generation and by how the machine is built.

## A common mistake {#mistake}
A common slip is reading a server's GPU count and ignoring how the GPUs are connected. Two machines with the same GPUs can differ greatly in speed if one links them with NVLink and the other only with PCIe. Ask about the links before you ask about the price.

## Key points {#key-points}
- The interconnect moves data between GPUs, and its bandwidth limits how you can split a model.
- PCIe is the standard slot connection, present everywhere but slower for GPU-to-GPU traffic.
- NVLink is NVIDIA's direct GPU link, many times faster than PCIe.
- Check the links before choosing tensor parallelism.
