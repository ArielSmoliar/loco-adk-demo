"""Visualize LOCO scheduling — run the mock demo and plot the results.

Generates 3 charts:
1. Timeline: who held the resource at each tick (Gantt-style)
2. Cost breakdown: per-agent spend
3. Contention: queue depth over time

Usage:
    python visualize.py
    python visualize.py --capacity 1   # heavy contention
    python visualize.py --capacity 5   # light contention
"""

from __future__ import annotations

import argparse
import asyncio
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from loco import AsyncLOCOScheduler, SharedResource
from loco.adapters.google_adk import ADKAdapter


TICKETS = [
    ("Hi, what are your business hours?", "simple"),
    ("Charged twice. Order #12345.", "complex"),
    ("How do I reset my password?", "simple"),
    ("App crashes on file upload.", "complex"),
    ("Thanks for quick delivery!", "simple"),
    ("Cancel my account. No response.", "complex"),
    ("Status of my order?", "simple"),
    ("Security vulnerability found.", "complex"),
]

# Colors by agent type
COLORS = {
    "triage": "#64B5F6",      # blue
    "support": "#81C784",     # green
    "escalation": "#E57373",  # red
}


def get_color(agent_name: str) -> str:
    for key, color in COLORS.items():
        if key in agent_name:
            return color
    return "#BDBDBD"


async def run_and_collect(capacity: int):
    """Run the mock demo and collect scheduling events."""
    scheduler = AsyncLOCOScheduler(
        [], SharedResource("gemini_api", capacity=capacity),
        optimize_for="balanced", auto_tune=True,
    )
    adapter = ADKAdapter(scheduler)

    # Collect events
    events = []  # (tick, agent_name, event_type, weight)
    grant_timeline = []  # (start_tick, end_tick, agent_name)

    original_submit = scheduler.submit_task

    async def tracked_submit(agent_id, task):
        await original_submit(agent_id, task)
        events.append({
            "tick": scheduler.logical_tick,
            "agent": agent_id,
            "event": "enqueue",
            "weight": task.weight,
        })

    scheduler.submit_task = tracked_submit

    # Process tickets
    async def process_ticket(ticket_id, ticket, classification):
        triage_name = f"triage-{ticket_id}"
        ctx = type("Ctx", (), {"agent_name": triage_name, "model": "gemini-2.0-flash"})()
        await adapter.before_model(ctx, None)
        start = scheduler.logical_tick
        await asyncio.sleep(random.uniform(0.05, 0.1))
        await adapter.after_model(ctx, classification)
        grant_timeline.append((start, scheduler.logical_tick, triage_name))

        if classification == "complex":
            name = f"escalation-{ticket_id}"
            model = "gemini-2.5-pro"
        else:
            name = f"support-{ticket_id}"
            model = "gemini-2.0-flash"

        ctx2 = type("Ctx", (), {"agent_name": name, "model": model})()
        await adapter.before_model(ctx2, None)
        start2 = scheduler.logical_tick
        await asyncio.sleep(random.uniform(0.1, 0.2))
        await adapter.after_model(ctx2, "response")
        grant_timeline.append((start2, scheduler.logical_tick, name))

    random.seed(42)
    await asyncio.gather(*(
        process_ticket(i, t, c) for i, (t, c) in enumerate(TICKETS)
    ))

    return scheduler, events, grant_timeline


def plot_results(scheduler, events, grant_timeline, capacity):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"LOCO-Agent Scheduling Visualization — {len(TICKETS)} tickets, "
        f"capacity={capacity}",
        fontsize=14, fontweight="bold",
    )

    # --- Chart 1: Timeline (Gantt) ---
    ax = axes[0]
    agents_seen = []
    for start, end, name in grant_timeline:
        if name not in agents_seen:
            agents_seen.append(name)

    # Sort: triage first, then support, then escalation
    def sort_key(name):
        if "triage" in name:
            return (0, name)
        if "support" in name:
            return (1, name)
        return (2, name)
    agents_seen.sort(key=sort_key)

    y_map = {name: i for i, name in enumerate(agents_seen)}

    for start, end, name in grant_timeline:
        y = y_map[name]
        duration = max(end - start, 0.3)  # min visible width
        ax.barh(y, duration, left=start, height=0.6,
                color=get_color(name), edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(agents_seen)))
    ax.set_yticklabels(agents_seen, fontsize=7)
    ax.set_xlabel("Logical Tick")
    ax.set_title("Resource Timeline\n(who held the slot when)")
    ax.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    # --- Chart 2: Cost per agent type ---
    ax = axes[1]
    type_costs = {"triage": 0, "support": 0, "escalation": 0}
    for aid in scheduler.agents:
        cost = scheduler.metrics.agent_cost(aid)
        for t in type_costs:
            if t in aid:
                type_costs[t] += cost
                break

    bars = ax.bar(
        list(type_costs.keys()),
        list(type_costs.values()),
        color=[COLORS[t] for t in type_costs],
        edgecolor="white",
    )
    for bar, val in zip(bars, type_costs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Total Weight (cost proxy)")
    ax.set_title("Cost by Agent Type\n(higher = more expensive model)")

    # --- Chart 3: Agent count and contention ---
    ax = axes[2]
    total_agents = len(scheduler.agents)
    total_calls = sum(len(a.completed_tasks) for a in scheduler.agents.values())
    total_cost = scheduler.metrics.total_cost()
    ticks = scheduler.logical_tick
    alpha = scheduler.alpha

    stats = {
        "Agents": total_agents,
        "API calls": total_calls,
        "Logical ticks": ticks,
        "Capacity": capacity,
    }
    stats_text = "\n".join(f"{k}: {v}" for k, v in stats.items())
    stats_text += f"\n\nTotal cost: {total_cost:.0f}"
    stats_text += f"\nAlpha: {alpha:.3f}"
    stats_text += f"\n(auto-tuned from 0.250)"

    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=13, verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Summary")

    plt.tight_layout()
    output = f"loco_scheduling_capacity{capacity}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


async def main(capacity: int):
    scheduler, events, timeline = await run_and_collect(capacity)
    plot_results(scheduler, events, timeline, capacity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.capacity))
