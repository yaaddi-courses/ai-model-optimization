# Cost per request: renting an API or running your own

**Covers:** Cost per request, Cost-per-request, Right-sizing, Per-token pricing, Break-even point, Self-hosting, API

## Putting a price on every answer {#overview}
Speed and size are means. The reason a team optimizes is usually money, and money is easiest to compare as a cost for each answer. This article covers how to compute that number, the way providers charge for their models, and when it pays to run your own.

## Cost per request {#cost-per-request}
**Cost per request** is the money spent on serving divided by the number of requests handled in the same period. It also appears hyphenated, as cost-per-request. One number lets you compare an optimization, a hardware choice or a provider against another. If a GPU costs 2.50 dollars an hour and serves 20 requests per second, the cost is 2.50 divided by 72,000 requests per hour, which is about 0.00003 dollars each.

## API {#api}
An **API**, or application programming interface, is a defined way for one program to ask another for something. A model provider sells access to its models through an API: your program sends a prompt, and the provider sends back the reply. You do not see or manage any hardware.

## Per-token pricing {#per-token-pricing}
**Per-token pricing** bills a model by the tokens you send in and the tokens you get back, often quoted per million tokens. Output tokens usually cost more than input tokens. Longer prompts and longer answers cost proportionally more, so trimming a prompt saves money directly.

## Self-hosting {#self-hosting}
**Self-hosting** means running a model on hardware you rent or own, not calling a provider's API. It gives you control over the model, its data and its speed, and it can cost less at high volume. It also means you take on operations work: deployment, scaling, monitoring and failures.

## Right-sizing {#right-sizing}
**Right-sizing** means choosing hardware just big enough for your workload instead of overprovisioning. A large GPU running at 10% use wastes most of its price. A smaller GPU, or a smaller model on the same one, may serve the same traffic for far less.

## Break-even point {#break-even-point}
The **break-even point** is the traffic level at which two options cost the same. Below it, the option with the lower fixed cost wins. Above it, the option with the lower cost for each request wins. For API versus self-hosting, an API has no fixed cost but a price for every request, while a rented GPU has a fixed monthly cost and a tiny cost for each request.

## A worked example {#example}
A rented GPU costs 2.50 dollars an hour. An API charges 1 dollar per million input tokens and 3 dollars per million output tokens. A typical request has 1,000 input tokens and 300 output tokens.

```python
gpu_per_hour = 2.50
hours_per_month = 730
gpu_monthly = gpu_per_hour * hours_per_month
print(gpu_monthly)                                 # 1825.0

api_per_request = 1000 / 1e6 * 1.0 + 300 / 1e6 * 3.0
print(round(api_per_request, 4))                   # 0.0019

break_even = gpu_monthly / api_per_request
print(round(break_even))                           # 960526

requests_per_second = 20
print(round(gpu_per_hour / (requests_per_second * 3600), 6))   # 3.5e-05
```

How to read it:

1. The GPU costs 1,825 dollars a month, no matter how busy it is.
2. The API costs 0.0019 dollars per request, 0.19 cents.
3. Dividing gives the break-even point: about 960,000 requests a month, or roughly 1,300 an hour.
4. Below that volume the API is cheaper. Above it, the GPU wins, as long as it can handle the load and someone looks after it.

The last line shows why: a busy GPU costs a tiny fraction of a cent per request.

## A common mistake {#mistake}
Counting only the hardware bill when comparing self-hosting with an API. Add the time of the people who run it, monitoring, spare capacity for failures and the idle hours at night. These can double the true cost.

## Key points {#key-points}
- Cost per request divides total spending by the number of requests, giving one comparable number.
- An API sells model access; per-token pricing charges for tokens in and out.
- Self-hosting trades a fixed cost and operations work for a low cost per request.
- The break-even point says at what volume the two options cost the same.
- Right-sizing avoids paying for hardware that stays idle.
