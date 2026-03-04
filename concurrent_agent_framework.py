import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ConcurrentBuilder, Message
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Load Azure OpenAI config from env
load_dotenv()
client = AzureOpenAIChatClient(credential=AzureCliCredential())

# Define two agents with different expertise for parallel answers
hr_agent = client.create_agent(
    instructions="You are an HR expert. Answer from the employee well-being perspective.",
    name="HR_Agent",
)
biz_agent = client.create_agent(
    instructions="You are a business operations expert. Answer from the perspective of company productivity and strategy.",
    name="Business_Agent",
)

# Custom aggregator: combine responses from both agents
def aggregate_responses(responses: list):
    combined = []
    for resp in responses:
        text = resp.agent_run_response.text if resp.agent_run_response else str(resp)
        combined.append(f"{resp.executor_id}: {text}")
    return "\n\n".join(combined)

# Build concurrent workflow with custom aggregator
workflow = ConcurrentBuilder().participants([hr_agent, biz_agent]).with_aggregator(aggregate_responses).build()

# Example question to be answered by both agents
question = "What are the biggest benefits and challenges of remote work for a company?"
async def run_workflow():
    result = await workflow.run(question)
    outputs = result.get_outputs()
    if outputs:
        print("===== Combined Responses =====")
        for item in outputs:
            if isinstance(item, list):
                for msg in item:
                    if hasattr(msg, 'role'):
                        speaker = msg.author_name or ("assistant" if msg.role == "assistant" else "user")
                        print(f"\n[{speaker}]:\n{msg.text}")
                    else:
                        print(f"\n{msg}")
            elif isinstance(item, str):
                print(f"\n{item}")
            else:
                print(f"\n{item}")

asyncio.run(run_workflow())
