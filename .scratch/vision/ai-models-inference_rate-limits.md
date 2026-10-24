> ## Documentation Index
> Fetch the complete documentation index at: https://docs.tokenfactory.nebius.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Rate Limits & Scaling

> Learn how rate limiting of requests works and what are the scaling rules

Nebius AI Inference is elastic by design. When you gradually increase your workload we scale the capacity with you. Most teams on Startup Tier never need to think about rate limits—but when you start pushing tens of thousands of requests per minute it helps to know how the platform behaves. Sudden bursts of AI Model traffic may be subject to throttling. This page explains the **dynamic rate‑limiting** system, how it auto‑scales, and the knobs you can turn when you need more headroom.

> [!TIP]
>
> **Need unlimited throughput?**\
> If you have sustained workloads that dwarf the standard limits, [reach out to us to upgrade to the Enterprise Tier](https://nebius.com/?_gl=1*1ni0ub9*_gcl_au*OTY2MTI0ODU5LjE3NTA4MzkxODkuOTQxMDE3MTg3LjE3NTEwMTgyMjYuMTc1MTAxODIyNg..#ai-contact-form). Enterprise removes the soft caps described below and gives you dedicated capacity plus an SLA.

### How it works

Key idea – Limits are dynamic. If your app stays close to the ceiling, we automatically raise it. If traffic falls off, we scale it back.

Every user is provided with a default rate limit cap. This limit grows automatically when you consistently operate close to your capacity and current limit.

> [!TIP]
>
> You can find the defaults at Rate Limits section of [Nebius Token Factory](https://tokenfactory.nebius.com/project/rate-limits)

We evaluate usage in rolling 15‑minute buckets:

* **Scale‑up rule** – When **average** usage in a 15‑minute window ≥ 80 % of the current limit, the limit for the *next* window increases by 20 % (`× 1.2`).
* **Scale‑down rule** – When average usage in a 15‑minute window ≤ 50 %, the limit for the next window decreases by one‑third (`÷ 1.5`).
* **Hard ceiling** – The limit can grow to **20 ×** your base allocation. Beyond that we require an Enterprise plan.

Visual example, assuming you continue to drive ≥ 80 % utilisation each window:

| **Window** | **Scale factor** | **RPM limit** | **TPM limit** |
| :- | :- | :- | :- |
| Baseline | `1.00×` | 60 | 400,000 |
| + 15 min | `1.20×` | 72 | 480,000 |
| + 30 min | `1.44×` | 86 | 576,000 |
| + 1 h | `2.07×` | 124 | 828,000 |
| + 2 h | `4.30×` | 258 | 1,720,000 |

### Monitoring your headroom and handling 429s

Your current rate limits are always visible in the API response headers, which are detailed further down.

If you exceed the active limit you will receive **HTTP 429**. To avoid this, you can track your consumption against your limits using the `x-ratelimit-remaining-requests` and `x-ratelimit-remaining-tokens` response headers.

> [!TIP]
>
> **Use Batch API for async workloads - it has significantly higher limits. Read more**

### **Over-the-Limit Processing**

In some cases, requests that exceed your rate limit may still be processed. This occurs when the system has spare capacity available, and your request can be handled with a lower priority without affecting other users.

These responses include:

```text theme={null}
x-ratelimit-over-limit: yes
```

This indicates that while your request was successful, you are over your nominal limit, and subsequent requests may be throttled if you continue to exceed your quota. Treat this as an early warning—subsequent calls may be throttled.

### Rate Limit Response Headers

Here's a simple explanation of response headers you can use to track you limits quotas:

**Description of Rate Limits response headers**

| Header | Description |
| :- | :- |
| `x-ratelimit-limit-requests` | The maximum number of requests that are permitted per minute. |
| `x-ratelimit-limit-tokens` | The maximum number of tokens that are permitted per minute. |
| `x-ratelimit-remaining-requests` | The remaining number of requests that are permitted before exhausting the rate limit. |
| `x-ratelimit-remaining-tokens` | The remaining number of tokens that are permitted before exhausting the rate limit. Note that the limit is replenished continuously. If your usage is sustainably below the rate limit, this number will hover near its maximum value. |
| `x-ratelimit-reset-requests` | The time in seconds until the rate limit (based on requests) resets. |
| `x-ratelimit-reset-tokens` | The time in seconds until the rate limit (based on tokens) resets. |
| `x-ratelimit-dynamic-scale-requests` | The current dynamic scale factor applied to your request rate limit. |
| `x-ratelimit-dynamic-scale-tokens` | The current dynamic scale factor applied to your token rate limit. |
| `x-ratelimit-dynamic-period-remaining` | The time in seconds until the next dynamic rate limit adjustment. |
| `x-ratelimit-dynamic-period-usage-requests` | Your average request usage in the current adjustment period, as a percentage. |
| `x-ratelimit-dynamic-period-usage-tokens` | Your average token usage in the current adjustment period, as a percentage. |
| `x-ratelimit-over-limit` | Set to "yes" if a request that is over the limit is processed. This happens when the system has spare capacity, and the request is handled with a lower priority. |
| `Retry-After` | The time in seconds to wait before making another request if the rate limit is exceeded. |
