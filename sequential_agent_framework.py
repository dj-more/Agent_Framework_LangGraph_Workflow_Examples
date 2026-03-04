import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import SequentialBuilder
from agent_framework import Message
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Load Azure OpenAI settings from .env (or environment variables)
load_dotenv()

# Initialize Azure OpenAI chat client (uses Azure AD credentials or API key from env)
client = AzureOpenAIChatClient(credential=AzureCliCredential())

# Define two agents for sequential pipeline:
# Agent1: Summarize a text into key bullet points
summarizer_agent = client.create_agent(
    instructions=(
        "You are a summarization assistant. Your task is to summarize the provided text into 2-3 key bullet points."
    ),
    name="summarizer_agent",
)
# Agent2: Provide an insight or conclusion based on the summary
insight_agent = client.create_agent(
    instructions=(
        "You are an insightful analyst. Read the summary (bullet points) of a text and provide one concise insight or conclusion based on it."
    ),
    name="insight_agent",
)

# Build sequential workflow: summarizer_agent -> insight_agent
workflow = SequentialBuilder().participants([summarizer_agent, insight_agent]).build()

# Example input: a short report text to summarize
report_text = (
    "AcmeCorp's Q4 report shows revenue grew by 20% year-on-year to $5M, "
    "with strong performance in the electronics division. However, net profit margin decreased slightly "
    "due to higher marketing expenses. The company plans to expand into new markets next year."
)
user_prompt = f"Summarize the following report:\n{report_text}"

# Run the workflow (asynchronous workflow execution)
async def run_workflow():
    result = await workflow.run(user_prompt)
    outputs = result.get_outputs()
    if outputs:
        final_conversation = outputs[-1]  # last agent's output messages
        print("===== Conversation =====")
        for msg in final_conversation:
            role = msg.role
            speaker = msg.author_name or ("assistant" if role == "assistant" else "user")
            print(f"[{speaker}]: {msg.text}")

asyncio.run(run_workflow())
