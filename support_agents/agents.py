"""Three ADK agents for a customer support workflow.

Triage:     Reads the ticket, classifies urgency (fast, cheap model)
Support:    Drafts a response to the customer (medium model)
Escalation: Handles complex issues that need deep reasoning (expensive model)

All three hit Gemini through the same API key. Without LOCO, they'd
collide under load. With LOCO, urgent escalations get priority.
"""

from google.adk import Agent

triage_agent = Agent(
    name="triage",
    model="gemini-2.5-flash",
    instruction="""You are a customer support triage agent.
    Read the incoming ticket and classify it:
    - "simple": greeting, FAQ, status check → route to support agent
    - "complex": billing dispute, bug report, account issue → route to escalation agent

    Respond with ONLY the classification word: "simple" or "complex".""",
)

support_agent = Agent(
    name="support",
    model="gemini-2.5-flash",
    instruction="""You are a friendly customer support agent.
    Draft a helpful, concise response to the customer's issue.
    Keep responses under 3 sentences. Be warm and professional.""",
)

escalation_agent = Agent(
    name="escalation",
    model="gemini-2.5-pro",
    instruction="""You are a senior support escalation agent.
    Handle complex issues that require deep reasoning:
    billing disputes, technical bugs, account problems.
    Provide a thorough, detailed response with next steps.
    Reference specific policies where relevant.""",
)
