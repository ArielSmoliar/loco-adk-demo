"""LOCO scheduling layer for the support agents.

Wraps ADK agent invocations with LOCO acquire/release so that:
- All 3 agents share a bounded Gemini API pool
- Escalation (expensive model) gets priority when competing with triage
- No agent starves -- even cheap triage calls eventually get through
- Every call is logged with cost and scheduling metadata
"""

import logging

from loco import AsyncLOCOScheduler, SharedResource
from loco.adapters.google_adk import ADKAdapter

logger = logging.getLogger("loco.scheduler")


def create_scheduler(capacity: int = 3) -> tuple[AsyncLOCOScheduler, ADKAdapter]:
    """Create a LOCO scheduler and ADK adapter for the support system.

    Args:
        capacity: How many concurrent Gemini API calls to allow.
                  Match this to your API rate limit.

    Returns:
        (scheduler, adapter) tuple. Wire the adapter's callbacks
        into your ADK agents.
    """
    scheduler = AsyncLOCOScheduler(
        [],  # agents auto-register on first call
        SharedResource(name="gemini_api", capacity=capacity),
        optimize_for="balanced",
        auto_tune=True,  # alpha adjusts based on observed contention
    )

    adapter = ADKAdapter(scheduler)

    # Quiet the scheduler log by default; set LOCO_LOG=1 to see JSON events
    import os
    if os.environ.get("LOCO_LOG"):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.getLogger("loco.scheduler").setLevel(logging.WARNING)

    return scheduler, adapter
