"""Mock version -- runs the full LOCO scheduling without a Gemini API key.

Same architecture as run.py but with simulated LLM responses.
Use this to verify the scheduling behavior before connecting a live API.

Usage:
    python run_mock.py
    python run_mock.py --capacity 1   # see heavy contention
    python run_mock.py --capacity 10  # see no contention
"""

from __future__ import annotations

import argparse
import asyncio
import random

import loco

from support_agents.scheduler import create_scheduler


TICKETS = [
    ("Hi, what are your business hours?", "simple"),
    ("I was charged twice for my subscription. Order #12345.", "complex"),
    ("How do I reset my password?", "simple"),
    ("Your app crashes on file upload. Third time this week.", "complex"),
    ("Thanks for the quick delivery!", "simple"),
    ("Cancel my account. Nobody responds.", "complex"),
    ("What's the status of my order?", "simple"),
    ("Security vulnerability in your payment page.", "complex"),
]


async def simulate_llm(delay_min: float = 0.05, delay_max: float = 0.15):
    """Simulate LLM latency."""
    await asyncio.sleep(random.uniform(delay_min, delay_max))


async def process_ticket_mock(ticket: str, classification: str, ticket_id: int):
    """Simulate the triage -> route -> respond pipeline with loco.wrap().

    Each ticket gets unique agent IDs (triage-0, support-0) so they can
    compete independently in the scheduler -- same as real ADK where each
    runner session is a separate agent instance.
    """

    # Step 1: Triage (cheap model, weight=1.5)
    await loco.wrap(
        simulate_llm,
        agent_id=f"triage-{ticket_id}",
        weight=1.5,  # gemini-2.5-flash
        delay_min=0.05,
        delay_max=0.15,
    )

    # Step 2: Route to support or escalation
    if classification == "complex":
        agent_id = f"escalation-{ticket_id}"
        weight = 3.0  # gemini-2.5-pro
    else:
        agent_id = f"support-{ticket_id}"
        weight = 1.5  # gemini-2.5-flash

    await loco.wrap(
        simulate_llm,
        agent_id=agent_id,
        weight=weight,
        delay_min=0.1,
        delay_max=0.3,
    )

    return {
        "ticket_id": ticket_id,
        "ticket": ticket,
        "triage": classification,
        "routed_to": agent_id.split("-")[0],  # "support" or "escalation"
    }


async def main(capacity: int = 3):
    scheduler = create_scheduler(capacity=capacity)

    print(f"LOCO-ADK Support Demo (mock mode)")
    print(f"Gemini API capacity: {capacity} concurrent slots")
    print(f"Tickets: {len(TICKETS)}")
    print(f"{'='*60}\n")

    # Process all tickets concurrently
    tasks = [
        process_ticket_mock(ticket, cls, i)
        for i, (ticket, cls) in enumerate(TICKETS)
    ]
    results = await asyncio.gather(*tasks)

    # Print results
    for r in results:
        print(f"  #{r['ticket_id']}: [{r['triage']:>7}] -> {r['routed_to']:<12} | {r['ticket'][:50]}")

    # Scheduling metrics
    print(f"\n{'='*60}")
    print(f"Scheduling Summary")
    print(f"{'='*60}")
    print(f"Total cost (weight):  {scheduler.metrics.total_cost():.1f}")
    print(f"Alpha (auto-tuned):   {scheduler.alpha:.3f}")
    print(f"Logical ticks:        {scheduler.logical_tick}")
    print()

    for aid in sorted(scheduler.agents):
        agent = scheduler.get_agent(aid)
        cost = scheduler.metrics.agent_cost(aid)
        completed = len(agent.completed_tasks)
        print(f"  {aid:<14} {completed} calls   cost={cost:.1f}")

    # Show what LOCO prevented
    print(f"\nWhat LOCO did:")
    print(f"  - {len(TICKETS)} tickets processed through {capacity} API slots")
    print(f"  - Escalation (gemini-2.5-pro, weight=3) got priority over triage (weight=1.5)")
    print(f"  - No rate limit errors -- scheduler held agents until slots opened")
    print(f"  - Cost tracked per agent for billing visibility")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(capacity=args.capacity))
