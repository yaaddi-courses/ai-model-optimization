# Cold starts, warm pools and scaling to zero

**Covers:** Cold start, Cold-start, Warm pool, Scale-to-zero

## Why the first request is slow {#overview}
Adding a replica is not instant. A fresh machine has to start, load the software, copy a model of many gigabytes into GPU memory and get ready. Until that finishes it cannot answer anyone. If a request lands on a machine that is still preparing, or triggers the preparation itself, that user waits far longer than usual. Autoscaling saves money by removing idle capacity, and this delay is the price it can cost the next user.

## Cold start {#cold-start}
A **cold start** is the extra delay when a fresh instance must boot and load the model weights before serving its first request. For a large model it can last many seconds, or even minutes. The same idea appears in other places as a cold-start problem, and it is also written with a hyphen as in "cold-start latency". The delay does not affect later requests on the same instance, which is why averages hide it. It shows up in the slowest few percent of requests.

## Warm pool {#warm-pool}
A **warm pool** is a set of instances that are already started and loaded with the model, but are not yet serving traffic. When demand rises, new traffic is sent to a warm instance straight away, and users avoid the cold start. Keeping a warm pool costs money, because the machines are paid for while they wait. You trade a steady cost for a faster response to spikes.

## Scale-to-zero {#scale-to-zero}
**Scale-to-zero** shuts down every replica when there is no traffic, so nothing is billed. It is attractive for tools used a few times a day. The cost is that the next request after a quiet period meets a full cold start, and that user waits while the first replica boots.

## A worked example {#example}
A 14 GB model is loaded from fast storage at 1 GB per second, with 6 seconds of other start-up work.

```python
model_gb = 14
load_speed_gb_per_s = 1.0
other_startup_s = 6

cold_start = model_gb / load_speed_gb_per_s + other_startup_s
print(cold_start)                      # 20.0 seconds

# A warm pool of 2 idle machines at 2 dollars an hour, all day
idle_machines = 2
price_per_hour = 2.0
print(idle_machines * price_per_hour * 24)    # 96.0 dollars per day

# Scale-to-zero: pay nothing overnight, but the first request waits
requests_per_morning_start = 1
print(requests_per_morning_start * cold_start)     # 20.0 seconds of waiting for one user
```

How to read it:

1. Loading 14 GB at 1 GB/s takes 14 seconds, plus 6 seconds of other start-up work: a 20-second cold start.
2. Two idle warm machines cost 96 dollars a day, whether or not anyone uses them.
3. Scale-to-zero costs nothing at night, and the price is one user who waits 20 seconds in the morning.
4. Which is cheaper depends on how often it happens and how much a 20-second wait hurts.

Smaller or faster-loading models shrink the cold start, which is one more reason to shrink weights.

## A common mistake {#mistake}
Looking only at the average latency after enabling scale-to-zero. The average hardly moves, while a few users have a terrible experience. Track the slowest percentiles and the count of cold starts.

## Key points {#key-points}
- A cold start is the delay while a new instance boots and loads the model.
- A warm pool keeps loaded instances ready, trading steady cost for fast response to spikes.
- Scale-to-zero stops all billing at idle, and the next user pays with a cold start.
- Smaller weights and faster storage shorten the cold start.
