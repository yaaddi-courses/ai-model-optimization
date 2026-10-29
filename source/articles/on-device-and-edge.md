# On-device and edge deployment

**Covers:** On-device inference, Edge deployment, Thermal throttling, Core ML, Hybrid deployment, GPS

## Moving the model closer to the user {#overview}
So far we have pictured the model in a data centre, with users sending requests across the network. Running it somewhere else has real advantages. The reply does not wait for a network round trip. The service works without a connection. Private data never leaves the person's device, and you do not pay for a server on every use. The price is that the hardware is small and varied, so the model has to be small too.

## On-device inference {#on-device-inference}
**On-device inference** runs a model directly on the user's own phone, laptop or gadget, not on a remote server. It gives fast replies, offline use and data that stays private. The model must fit in the device's memory and run within its power and heat limits, which usually means a heavily shrunk one.

## Edge deployment {#edge-deployment}
**Edge deployment** is a wider idea. It runs models on devices or servers close to the user, not in a central data centre. The edge may be a shop's local server, a router or a factory machine. It cuts network delay and can keep data local.

## Thermal throttling {#thermal-throttling}
Small devices cannot shed heat as fast as servers can. **Thermal throttling** is when hardware slows itself down to avoid overheating. A phone running a heavy model may be quick for the first minute, and then drop to half speed. Short bursts of work suit a phone well. Sustained work does not, so test the speed over minutes, not seconds.

## Core ML {#core-ml}
**Core ML** is Apple's framework for running machine learning models on iPhones, iPads and Macs. You give it a converted model, and it decides how to split the work between the CPU, the GPU and the device's Neural Engine, which is Apple's neural processing unit. The app does not have to make that choice.

## Hybrid deployment {#hybrid-deployment}
Why choose only one place? In **hybrid deployment**, inference is split between the device and the cloud. For example, a small model on the phone answers easy requests at once, and hard ones go to a bigger model on a server. This balances speed, cost and quality, and it keeps working in a basic way when the network is down.

## GPS {#gps}
**GPS**, the Global Positioning System, is satellite navigation. It is not a chip that runs neural networks, and it appears in questions as a name that does not belong with the chips above. It matters for on-device work for a different reason: location is among the most private data on a phone, so doing the work on the device means the position need not leave it.

## A worked example {#example}
Let's watch a phone model slow down as it heats up. We compare the average speed over several minutes with the speed at the start.

```python
tokens_per_second_by_minute = [30, 30, 24, 18, 15, 15]

start_speed = tokens_per_second_by_minute[0]
average = sum(tokens_per_second_by_minute) / len(tokens_per_second_by_minute)
slowest = min(tokens_per_second_by_minute)

print(start_speed)                          # 30
print(average)                              # 22.0
print(slowest, slowest / start_speed)       # 15 0.5

def choose(confidence, threshold=0.8):
    return "on device" if confidence >= threshold else "cloud"

print(choose(0.95), choose(0.6))            # on device cloud
```

How to read it:

1. The first minute runs at 30 tokens per second.
2. By minute five, throttling has cut it to 15, which is half the starting speed.
3. The average over six minutes is 22. A benchmark that stopped after one minute would report 30 and mislead you.
4. The small function shows the hybrid idea: confident answers stay on the device, and uncertain ones go to the cloud.

## A common mistake {#mistake}
A common slip is testing on one new phone and shipping to everyone. Devices differ widely in memory, chips and cooling. Test on the oldest and weakest device you plan to support.

## Key points {#key-points}
- On-device inference gives speed, offline use and privacy, within the device's limits.
- Edge deployment puts models close to users, but not necessarily on their device.
- Thermal throttling slows hardware when it heats, so test sustained speed.
- Core ML is Apple's framework, and hybrid deployment splits work between device and cloud.
- GPS is satellite navigation, not a neural-network chip.
