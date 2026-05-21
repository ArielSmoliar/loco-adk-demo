"""LOCO scheduling layer for the support agents.

Uses the convenience API so all 3 agents share a bounded Gemini API pool
with automatic priority, anti-starvation, and per-agent cost tracking.
"""

import logging
import os

import loco


def create_scheduler(capacity: int = 3) -> loco.AsyncLOCOScheduler:
    """Create a LOCO scheduler for the support system.

    Args:
        capacity: How many concurrent Gemini API calls to allow.
                  Match this to your API rate limit.

    Returns:
        The configured scheduler. Use loco.wrap() to schedule calls.
    """
    scheduler = loco.configure(
        capacity=capacity,
        resource_name="gemini_api",
    )

    # Quiet the scheduler log by default; set LOCO_LOG=1 to see JSON events
    if os.environ.get("LOCO_LOG"):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.getLogger("loco.scheduler").setLevel(logging.WARNING)

    return scheduler
