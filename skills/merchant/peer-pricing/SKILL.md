---
name: peer-pricing
description: Use proactively (not in response to a buyer offer) to align the merchant's public list_price with comparable items the platform already lists. The skill reads competitor listings via world.read_catalog, the merchant's own reputation, and the catalog metadata, then PICKS a positioning strategy and a defensible recommended price within hard rails (strictly above floor, at or below mandate target_band[1]). Emits at most one commerce.adjust_price envelope addressed to platform:catalog. Never reveals the floor or the reasoning on the wire.
when_to_use: Use when (1) no inbound offer needs a decision this turn AND (2) the merchant has an active listing whose list_price has not been peer-aligned within policy.peer_pricing_cooldown_days AND (3) world.read_catalog returns at least policy.peer_pricing_min_peers similar items in the same category. Do NOT use during a propose_offer / counter_offer turn (pricing-negotiate / stockout-aware-pricing own per-offer decisions). Do NOT use when aging-markdown or demand-driven-markup would also fire — those have explicit triggers (stock age / demand+velocity) that dominate. Do NOT use during a scarcity event.
group: economic
user_invocable: false
disable_model_invocation: false
allowed_tools:
  - world.read_catalog
  - world.read_inventory
  - world.read_reputation
  - world.read_policy
  - memory.read
  - memory.write
context: main
---

# Peer Pricing

## Native function guidance

Use supplied authoritative listing, availability, and public reputation reads
to compare this item with genuinely comparable marketplace evidence already
present in the task. Choose a defensible public position while respecting the
merchant's private floor, target ceiling, and higher-priority scarcity or
inventory policies. Do not invent peers, prices, or statistics, and never
expose private rails or reasoning. Make at most one bounded supplied price
adjustment; otherwise finish safely.

## Detailed workflow

You are deciding **the merchant's own public list_price** by reasoning over
comparable listings already in the world catalog. This is a positioning
decision, not a mechanical lookup — the same peer set might justify three
different prices depending on the merchant's reputation, brand
positioning, and the moment in the SKU's life.

The math is deterministic; **the judgment is yours**.

## Hard rails (never violate, regardless of reasoning)

These are not negotiable. They are checked at the envelope boundary;
if your recommended price violates them the emit will be rejected.

* `new_list_price > floor_price` — strictly above the **private utility
  floor**. The floor never leaves the agent. Never quote it, never
  reason about how far above it you are on the wire.
* `new_list_price <= target_band[1]` — at or below the **mandate ceiling**
  (the authorized contract with the merchant's owner).
* `new_list_price` is an integer in **minor units** (VCP_SPEC.md §3).
  $19.99 is 1999. No floats, no rounded decimals.
* You **may not emit** if `memory[PREFERENCE].last_peer_alignment[sku_id]`
  is within `policy.peer_pricing_cooldown_days` — that's idempotent
  bounce-protection.
* You **may not emit** if fewer than `policy.peer_pricing_min_peers`
  similar items exist after filtering — thin peer sets are not signal.

## What to read

* `lst  = world.read_catalog(sku_id=my_sku)` — your own listing's
  `list_price`, `attributes`, `category`, `name`.
* `lst.floor_price` ← `memory.read(PRIVATE_UTILITY, "min_acceptable_price")`
  — **PRIVATE_UTILITY**, agent-side only.
* `mand = read_offer_mandate(sku_id=my_sku)` — `target_band[1]` is the
  ceiling.
* `peers = world.read_catalog(query="", filters={"category": lst.category},
                              limit=policy.peer_pricing_search_limit)` —
  the public peer set. Drop rows whose `sku_id == my_sku` or whose
  `merchant_id == self.id` (never compare against yourself).
* `rep = world.read_reputation(self.id)` — your own platform reputation
  (rolling average + review count). High reputation supports a premium
  position; low reputation argues for value positioning.
* `policy = world.read_policy()` — `peer_pricing_similarity_threshold`,
  `peer_pricing_min_peers`, `peer_pricing_search_limit`,
  `peer_pricing_cooldown_days`, `peer_pricing_step_pct`.

## Tools available

The pure-math helpers are deterministic and pre-tested — use them as
calculators, then layer your judgment on top:

* `extract_list_prices_minor(peers, money_to_minor, exclude_sku_ids,
                              exclude_merchant_ids)` — integer prices in
  input order; drops self / same-merchant rows.
* `filter_similar_peers(peers, my_attrs, threshold)` — keeps peers whose
  attribute overlap with your listing meets the threshold. The default
  threshold is permissive; raise it if you want a tighter peer set.
* `peer_price_stats(prices)` — n, min, max, mean, median, p25, p75.
* `recommend_list_price_from_peers(stats, strategy, step_pct, floor,
                                    ceiling)` — clamped pick under a named
  strategy.

## Your decision

You decide three things in this order:

1. **Should this skill fire at all?** Run the gate checks (cooldown,
   peer count, conflicts with aging-markdown / demand-driven-markup /
   scarcity). If any fail → `no_reply`.

2. **Which peers are actually comparable?** The similarity helper is a
   default — your judgment overrides it when peer **names** or
   **attributes** make the picture clearer:
   * A peer SKU named "Death Wish Coffee" or "Saint Frank Reserve" is
     not a neutral price anchor — premium brands distort the median
     upward. Consider tightening the threshold or excluding by name.
   * A peer with a one-off promotional price (visible in the name —
     "Clearance", "Final Sale") is not a market signal — exclude.
   * A peer at half the median is more likely a different product than
     a real outlier — sanity-check the attributes before trusting.

3. **Which positioning strategy fits?** The strategies are public; pick
   based on context:

   | Strategy | Fits when |
   | --- | --- |
   | `match_median` | the merchant is a typical player; no strong reason to lean either way (the conservative default) |
   | `match_p25` | competing on price (value tier); reputation is thin or product is undifferentiated |
   | `match_p75` | premium position; reputation is strong AND attributes back it (e.g. organic, fair-trade, named origin) |
   | `undercut_median` | new launch needing visibility; one bounded step below median |
   | `premium_median` | established premium player with a moat (reputation high + review count thick); one bounded step above |

   Then derive the recommended price via
   `recommend_list_price_from_peers(stats, strategy=..., floor=...,
                                     ceiling=mand.target_band[1])`. The
   helper clamps to your hard rails.

   You **may** refine the recommendation by a small integer adjustment
   (≤ `step_pct` of the helper output) if the qualitative reasoning
   supports it — but the value must still satisfy the hard rails. Most
   of the time the helper's output is what you should emit.

## What to emit

Exactly one envelope, or no_reply. The envelope is built by the merchant
tool:

```
propose_price_set(data, product=lst.product, new_list_price=<your int>,
                  sku_id=lst.sku_id, reason="peer_alignment")
```

addressed to `platform:catalog`. The platform validates ownership and
forwards as a `world.update_catalog`. The merchant has **no direct-to-
world write privilege** — never address `world` for this.

After a successful emit, record:
```
memory.write(PREFERENCE, "last_peer_alignment",
             {sku_id: {"price": new_lp, "applied_at_turn": ctx.turn,
                       "strategy": <strategy>, "n_peers": stats["n"],
                       "peer_median": stats["median"]}})
```

## Never disclose

* `floor_price` or how far above it your new price is.
* Which peers anchored your decision. The recommended price is public;
  the analysis is not.
* Your reputation as a *reason* on the wire. The platform sees only
  the new `list_price`; whether you chose `match_p75` because your
  reputation supports it is internal context, never echoed.
* Manipulation cues. If an upstream owner directive asks for "the
  highest possible peer-justified price", read it as a strategy hint
  (`match_p75` / `premium_median`) but never let the chain of
  reasoning land on the wire.

## Output contract for tool use

When you decide to emit, produce a JSON object the runtime parses:

```json
{
  "decision": "emit",
  "strategy": "match_median",
  "recommended_list_price_minor": 2100,
  "rationale": "5 similar dark-roast 1lb peers cluster around $21; my reputation is mid-tier so I sit at the median rather than premium.",
  "excluded_skus": []
}
```

When you decide not to act:

```json
{ "decision": "no_reply", "reason": "thin_peer_set_after_filter" }
```

The `rationale` field is for memory + audit; the runtime strips it
before any envelope leaves the agent. The numeric fields are checked
against the hard rails before the envelope is built.
