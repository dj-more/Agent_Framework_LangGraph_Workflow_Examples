import asyncio
import logging
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import MagenticBuilder
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Suppress expected cycle/round-limit warnings from agent_framework internals
logging.getLogger("agent_framework").setLevel(logging.CRITICAL)

# Load config and initialize Azure Chat client
load_dotenv()
client = AzureOpenAIChatClient(credential=AzureCliCredential())

# Define two agents with opposing viewpoints
pro_agent = client.create_agent(
    instructions="You strongly argue in favor of the given topic. Keep responses concise (2-3 sentences).",
    name="ProAgent",
)
con_agent = client.create_agent(
    instructions="You strongly argue against the given topic. Keep responses concise (2-3 sentences).",
    name="ConAgent",
)

# Topic for the debate
topic_question = "Should AI be strictly regulated by the government?"

# Collect agent messages via event callback
debate_messages = []

async def capture_event(evt):
    """Capture agent messages as they occur during the debate."""
    from agent_framework import MagenticAgentMessageEvent
    if isinstance(evt, MagenticAgentMessageEvent):
        debate_messages.append((evt.agent_id, evt.message))

# Build a group chat workflow using MagenticBuilder with a standard manager
# The manager orchestrates turns between the agents (limited to 4 rounds)
workflow = (
    MagenticBuilder()
    .participants(ProAgent=pro_agent, ConAgent=con_agent)
    .with_standard_manager(
        chat_client=client,
        max_round_count=4,
        instructions="You are a debate moderator. Alternate between ProAgent and ConAgent for a balanced debate.",
    )
    .on_event(capture_event)
    .build()
)

async def run_workflow():
    await workflow.run(topic_question)
    print("===== Debate Transcript =====")
    for agent_id, msg in debate_messages:
        print(f"\n[{agent_id}]: {msg.text}")

asyncio.run(run_workflow())
