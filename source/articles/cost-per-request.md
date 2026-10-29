# Cost per request: renting an API or running your own

**Covers:** Cost per request, Cost-per-request, Right-sizing, Per-token pricing, Break-even point, Self-hosting, API

## Putting a price on every answer {#overview}
Speed and size are not the goal in themselves. A team usually optimizes to save money, and the easiest way to compare money is as a cost for each answer. In this article we compute that number, look at how providers charge for their models, and work out when it pays to run your own.

## Cost per request {#cost-per-request}
**Cost per request** is the money spent on serving, divided by the number of requests handled in the same period. It also appears hyphenated, as cost-per-request. Having one number lets you compare an optimization, a hardware choice or a provider against another. Suppose a GPU costs 2.50 dollars an hour and serves 20 requests per second. That is 72,000 requests an hour, so each one costs 2.50 divided by 72,000, which is about 0.00003 dollars.

## API {#api}
How do most people use a model without owning one? Through an **API**, short for application programming interface: an agreed way for one program to ask another for something. A model provider sells access to its models through an API. Your program sends a prompt, and the provider sends back the reply. You never see or manage any hardware.

## Per-token pricing {#per-token-pricing}
Providers usually charge by the amount of text. With **per-token pricing**, you pay for the tokens you send in and the tokens you get back, often quoted per million tokens. Output tokens usually cost more than input tokens. Longer prompts and longer answers cost proportionally more, so trimming a prompt saves money directly.

## Self-hosting {#self-hosting}
The other route is **self-hosting**: running a model on hardware you rent or own, instead of calling a provider's API. It gives you control over the model, its data and its speed, and at high volume it can cost less. It also means you take on the operations work: deployment, scaling, monitoring and handling failures.

## Right-sizing {#right-sizing}
Whichever route you choose, avoid paying for more than you need. **Right-sizing** means choosing hardware only as big as for your workload. A large GPU running at 10% use wastes most of its price. A smaller GPU, or a smaller model on the same one, may serve the same traffic for far less.

## Break-even point {#break-even-point}
When does self-hosting start to win? The **break-even point** is the traffic level at which two options cost the same. Below it, the option with the lower fixed cost wins. Above it, the option with the lower cost per request wins. For an API against self-hosting, the API has no fixed cost but a price on every request, while a rented GPU has a fixed monthly cost and a tiny cost on each request.

## A worked example {#example}
Let's find the break-even point. A rented GPU costs 2.50 dollars an hour. An API charges 1 dollar per million input tokens and 3 dollars per million output tokens. A typical request has 1,000 input tokens and 300 output tokens.

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

Step by step:

1. The GPU costs 1,825 dollars a month, no matter how busy it is.
2. The API costs 0.0019 dollars per request, which is 0.19 cents.
3. Dividing 1,825 by 0.0019 gives the break-even point: about 960,000 requests a month, or roughly 1,300 an hour.
4. Below that volume the API is cheaper. Above it, the GPU wins, as long as it can handle the load and someone looks after it.

The last line of the code shows why a busy GPU wins: each request costs a tiny fraction of a cent.

## A common mistake {#mistake}
A common slip is counting only the hardware bill when comparing self-hosting with an API. The hardware is the visible part. Add the time of the people who run it, the monitoring, the spare capacity for failures and the idle hours at night. Together these can double the true cost.

## Key points {#key-points}
- Cost per request divides total spending by the number of requests, giving one number you can compare.
- An API sells model access, and per-token pricing charges for tokens in and out.
- Self-hosting trades a fixed cost and operations work for a low cost per request.
- The break-even point tells you at what volume the two options cost the same.
- Right-sizing avoids paying for hardware that stays idle.
