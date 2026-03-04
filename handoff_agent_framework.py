import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import SequentialBuilder
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
client = AzureOpenAIChatClient(credential=AzureCliCredential())

# Define a general support agent that classifies and handles general queries
support_agent = client.create_agent(
    instructions=(
        "You are a customer support assistant. If the user is asking about a refund or return, "
        "say: 'Let me transfer you to our refund specialist.' and then briefly describe the issue. "
        "Otherwise, answer the general inquiry directly."
    ),
    name="SupportAgent",
)
# Define a specialist refund agent
refund_agent = client.create_agent(
    instructions="You are a customer service agent specialized in refund and return issues. Help the user with refund requests. Provide clear steps for the refund process.",
    name="RefundAgent",
)

# Build a sequential workflow: SupportAgent triages, then RefundAgent handles refund details
workflow = SequentialBuilder().participants([support_agent, refund_agent]).build()

# Simulate a user query about a refund to trigger the handoff
user_query = "I want to return my product and get a refund."

async def run_workflow():
    result = await workflow.run(user_query)
    outputs = result.get_outputs()
    if outputs:
        final_conv = outputs[-1]
        print("===== Handoff Conversation =====")
        for msg in final_conv:
            if hasattr(msg, 'text'):
                speaker = getattr(msg, 'author_name', None) or msg.role
                print(f"\n[{speaker}]: {msg.text}")

asyncio.run(run_workflow())
