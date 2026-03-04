import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import MagenticBuilder
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Load Azure OpenAI configuration
load_dotenv()
client = AzureOpenAIChatClient(credential=AzureCliCredential())

# Define two worker agents
research_agent = client.create_agent(
    instructions="You are a researcher agent. When given a task, you find relevant information and provide facts or data.",
    name="ResearchAgent",
)
analysis_agent = client.create_agent(
    instructions="You are an analysis agent. When given information, you provide insights, explanations, or conclusions.",
    name="AnalysisAgent",
)

# Build a magentic workflow with a standard manager as orchestrator
# The manager automatically plans and assigns tasks to the worker agents
workflow = (
    MagenticBuilder()
    .participants(ResearchAgent=research_agent, AnalysisAgent=analysis_agent)
    .with_standard_manager(
        chat_client=client,
        max_round_count=5,
        instructions=(
            "You are a planning agent. Break the user's goal into tasks and assign them to the appropriate worker agents. "
            "Available agents: ResearchAgent (find information), AnalysisAgent (analyze information)."
        ),
    )
    .build()
)

# Complex user goal
user_goal = "Find the latest GDP of France and analyze what it means for the European economy."

async def run_workflow():
    result = await workflow.run(user_goal)
    outputs = result.get_outputs()
    print("===== Task-Oriented Conversation =====")
    for output in outputs:
        if isinstance(output, list):
            for msg in output:
                if hasattr(msg, 'text'):
                    speaker = getattr(msg, 'author_name', None) or msg.role
                    print(f"\n[{speaker}]: {msg.text}")
        elif hasattr(output, 'text'):
            speaker = getattr(output, 'author_name', None) or 'unknown'
            print(f"\n[{speaker}]: {output.text}")
        else:
            print(f"\n{output}")

asyncio.run(run_workflow())
