"""List prices in US dollars per million tokens (input, output).

These were written down by hand when the code was written and are NOT fetched
from the providers. Check the providers' pricing pages before trusting a cost
estimate, or pass --price-in / --price-out to the eval script.
"""

from __future__ import annotations

PRICES_PER_MTOK: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-opus-5-5"): (4.00, 20.00),
    ("anthropic", "claude-sonnet-5-5"): (2.00, 10.00),
    ("anthropic", "claude-haiku-5-5"): (0.10, 0.50),  # prompts up to 100k tokens, which every email here is
    ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),
    ("anthropic", "claude-sonnet-4-5"): (3.00, 15.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4.1-mini"): (0.40, 1.60),
    ("fake", "fake-rules"): (0.0, 0.0),
}


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    return PRICES_PER_MTOK.get((provider, model))


def cost_usd(input_tokens: int, output_tokens: int, price: tuple[float, float] | None) -> float | None:
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
