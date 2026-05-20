"""Run the LOCO-scheduled customer support system with live Gemini API.

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

from google import genai
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


def make_user_content(text: str) -> genai.types.Content:
    """Create a user message Content object for ADK v2."""
    return genai.types.Content(
        role="user",
        parts=[genai.types.Part(text=text)],
    )


async def run_agent(runner: Runner, user_id: str, session_id: str, message: str) -> str:
    """Run an ADK agent and collect the text response."""
    text_parts = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=make_user_content(message),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
    return "".join(text_parts)


async def process_ticket(
    triage_runner: Runner,
    support_runner: Runner,
    escalation_runner: Runner,
    adapter,
    ticket: str,
    ticket_id: int,
) -> dict:
    """Process one ticket through the triage -> route -> respond pipeline."""

    user_id = f"customer-{ticket_id}"
    session_id = f"session-{ticket_id}"

    # Step 1: Triage -- classify the ticket
    triage_ctx = type("Ctx", (), {"agent_name": f"triage-{ticket_id}", "model": "gemini-2.5-flash"})()
    await adapter.before_model(triage_ctx, None)

    triage_text = await run_agent(triage_runner, user_id, f"{session_id}-triage", f"Classify this ticket: {ticket}")

    await adapter.after_model(triage_ctx, triage_text)

    # Step 2: Route to support or escalation based on triage
    is_complex = "complex" in triage_text.lower()
    target_runner = escalation_runner if is_complex else support_runner
    target_name = "escalation" if is_complex else "support"
    target_model = "gemini-2.5-pro" if is_complex else "gemini-2.5-flash"

    route_ctx = type("Ctx", (), {"agent_name": f"{target_name}-{ticket_id}", "model": target_model})()
    await adapter.before_model(route_ctx, None)

    response_text = await run_agent(target_runner, user_id, f"{session_id}-{target_name}", ticket)

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

    # Create one runner per agent type (ADK v2 pattern)
    triage_runner = Runner(
        agent=triage_agent,
        app_name="loco-support-demo",
        session_service=session_service,
        auto_create_session=True,
    )
    support_runner = Runner(
        agent=support_agent,
        app_name="loco-support-demo",
        session_service=session_service,
        auto_create_session=True,
    )
    escalation_runner = Runner(
        agent=escalation_agent,
        app_name="loco-support-demo",
        session_service=session_service,
        auto_create_session=True,
    )

    print(f"LOCO-ADK Support Demo (live Gemini API)")
    print(f"Gemini API capacity: {capacity} concurrent slots")
    print(f"Tickets: {len(TICKETS)}")
    print(f"{'='*60}\n")

    # Process all tickets concurrently -- LOCO handles contention
    tasks = []
    for i, ticket in enumerate(TICKETS):
        tasks.append(process_ticket(
            triage_runner, support_runner, escalation_runner,
            adapter, ticket, i,
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print results
    for r in results:
        if isinstance(r, Exception):
            print(f"ERROR: {r}\n")
            continue
        print(f"Ticket:    {r['ticket'][:60]}")
        print(f"Triage:    {r['triage'][:80]}")
        print(f"Routed to: {r['routed_to']}")
        print(f"Response:  {r['response'][:120]}...")
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
