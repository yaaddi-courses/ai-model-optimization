# Autoscaling: matching capacity to traffic

**Covers:** Autoscaling, Replica, Scale out, Queue depth, Minimum replicas

## What problem does this solve? {#overview}
Think of a restaurant. It is quiet at 10 a.m., full at lunch, and sometimes a crowd arrives without warning. Traffic to a model service behaves the same way: quiet at night, busy after lunch, spiky when a link goes viral. If you run exactly enough machines for the busiest minute, you pay for idle machines the rest of the day. If you run enough for the average, users wait at the peak.

**Autoscaling** is the answer. It lets the system change its own size: it watches a signal, adds capacity when demand rises, and removes it when demand falls.

## Replica {#replica}
What does "adding capacity" mean in practice? You start another copy of the server. A **replica** is one running copy of your model server. Each replica loads the model and answers requests on its own. A load balancer in front of them hands every new request to a replica that has room.

Replicas are interchangeable, and that is what makes them easy to add or remove. One replica serving 10 requests per second becomes three replicas serving about 30, as long as each has its own GPU.

## Scale out {#scale-out}
There are two ways to get more capacity. You can make one machine bigger, or you can run more machines. **Scaling out** is the second way: adding replicas side by side.

For model serving, scaling out is usually the better fit. A bigger GPU costs much more than twice as much, and there is a limit to how big one machine can get. More replicas also keep the service running if one of them fails.

## Queue depth {#queue-depth}
To decide when to add a replica, the autoscaler needs a signal. A popular one is **queue depth**: the number of requests waiting for a replica to pick them up.

Read it like a line at a shop. A queue of zero or one means the replicas are keeping up. A queue that keeps growing means users are waiting, and that is the moment to scale out. Queue depth follows real demand, unlike CPU or GPU use, which can look fine while requests pile up.

## Minimum replicas {#minimum-replicas}
Scaling down saves money, but it has a cost of its own. A new replica must start and load the model before it can help, and that takes time. **Minimum replicas** is a floor: the number of copies that always stay running, even at 3 a.m. with no traffic.

Set it to 1 or 2, and the first user of the morning still gets a fast answer. Set it to 0, and you pay nothing overnight, but someone has to wait. The right number depends on how much a slow first request costs you compared with an idle machine.

## A worked example {#example}
Suppose a service handles 20 requests per second per replica. The policy is to keep the queue short and never to go below 2 replicas.

```python
import math

per_replica = 20
minimum = 2

def replicas_needed(requests_per_second):
    return max(minimum, math.ceil(requests_per_second / per_replica))

for load in [5, 35, 90, 210]:
    print(load, replicas_needed(load))
# 5 2
# 35 2
# 90 5
# 210 11
```

We read the output together:

1. At 5 requests per second one replica would do, but the floor keeps 2.
2. At 35, you need 35 ÷ 20 = 1.75, which rounds up to 2.
3. At 90, 90 ÷ 20 = 4.5 rounds up to 5.
4. At 210, 210 ÷ 20 = 10.5 rounds up to 11.

We round up because half a replica cannot serve anyone. A real autoscaler also adds a delay between decisions, so it does not add and remove replicas every few seconds.

## A common mistake {#mistake}
A common slip is scaling on one quick measurement. Traffic is noisy, and each new replica takes time to start, so the fleet jumps up and down. Smooth the signal over a minute or two, and scale down more slowly than you scale up.

## Key points {#key-points}
- Autoscaling adds and removes capacity automatically as demand changes.
- A replica is one running copy of the server, and replicas share the traffic.
- Scaling out means more copies side by side, not a bigger single machine.
- Queue depth, the number of waiting requests, is a reliable signal for adding replicas.
- Minimum replicas keeps some capacity always ready, trading idle cost for a fast first answer.
