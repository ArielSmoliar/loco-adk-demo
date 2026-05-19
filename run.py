"""Run the LOCO-scheduled customer support system.

Sends a batch of customer tickets through 3 ADK agents (triage, support,
escalation), all sharing a bounded Gemini API pool via LOCO scheduling.

Usage:
    export GOOGLE_API_KEY="your-key"
    python run.py

    # Or with custom capacity:
    python run.py --capacity 5
"""

from __future__ import annotations

import argparse
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from support_agents.agents import triage_agent, support_agent, escalation_agent
from support_agents.scheduler import create_scheduler


# Sample customer tickets
TICKETS = [
    "Hi, what are your business hours?",
    "I was charged twice for my subscription last month. Order #12345.",
    "How do I reset my password?",
    "Your app crashes every time I try to upload a file. This is the third time.",
    "Thanks for the quick delivery!",
    "I need to cancel my account. I've been trying for a week and nobody responds.",
    "What's the status of my order?",
    "I found a security vulnerability in your payment page. Here are the details...",
]


async def process_ticket(
    runner: Runner,
    adapter,
    ticket: str,
    session_id: str,
) -> dict:
    """Process one ticket through the triage → route → respond pipeline."""

    # Step 1: Triage — classify the ticket
    triage_ctx = type("Ctx", (), {"agent_name": "triage", "model": "gemini-2.0-flash"})()
    await adapter.before_model(triage_ctx, None)

    triage_response = await runner.run_async(
        agent=triage_agent,
        session_id=session_id,
        user_message=f"Classify this ticket: {ticket}",
    )
    triage_text = ""
    async for event in triage_response:
        if hasattr(event, "text"):
            triage_text += event.text

    await adapter.after_model(triage_ctx, triage_text)

    # Step 2: Route to support or escalation based on triage
    is_complex = "complex" in triage_text.lower()
    target_agent = escalation_agent if is_complex else support_agent
    target_name = "escalation" if is_complex else "support"
    target_model = "gemini-2.5-pro" if is_complex else "gemini-2.0-flash"

    route_ctx = type("Ctx", (), {"agent_name": target_name, "model": target_model})()
    await adapter.before_model(route_ctx, None)

    response = await runner.run_async(
        agent=target_agent,
        session_id=session_id,
        user_message=ticket,
    )
    response_text = ""
    async for event in response:
        if hasattr(event, "text"):
            response_text += event.text

    await adapter.after_model(route_ctx, response_text)

    return {
        "ticket": ticket,
        "triage": triage_text.strip(),
        "routed_to": target_name,
        "response": response_text.strip(),
    }


async def main(capacity: int = 3):
    scheduler, adapter = create_scheduler(capacity=capacity)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=triage_agent,  # default agent for the runner
        app_name="loco-support-demo",
        session_service=session_service,
    )

    print(f"LOCO-ADK Support Demo")
    print(f"Gemini API capacity: {capacity} concurrent slots")
    print(f"Tickets: {len(TICKETS)}")
    print(f"{'='*60}\n")

    # Process all tickets concurrently — LOCO handles contention
    tasks = []
    for i, ticket in enumerate(TICKETS):
        session = await session_service.create_session(
            app_name="loco-support-demo",
            user_id=f"customer-{i}",
        )
        tasks.append(process_ticket(runner, adapter, ticket, session.id))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print results
    for r in results:
        if isinstance(r, Exception):
            print(f"ERROR: {r}\n")
            continue
        print(f"Ticket:    {r['ticket'][:60]}")
        print(f"Triage:    {r['triage']}")
        print(f"Routed to: {r['routed_to']}")
        print(f"Response:  {r['response'][:100]}...")
        print()

    # Scheduling metrics
    print(f"{'='*60}")
    print(f"Scheduling Summary")
    print(f"{'='*60}")
    print(f"Total cost (weight): {scheduler.metrics.total_cost():.1f}")
    print(f"Alpha: {scheduler.alpha:.3f} (auto-tuned)")
    for aid in sorted(scheduler.agents):
        agent = scheduler.get_agent(aid)
        cost = scheduler.metrics.agent_cost(aid)
        print(f"  {aid}: {len(agent.completed_tasks)} calls, cost={cost:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=3,
                        help="Gemini API concurrent slots (default: 3)")
    args = parser.parse_args()
    asyncio.run(main(capacity=args.capacity))
