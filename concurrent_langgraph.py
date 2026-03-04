from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

# Load Azure OpenAI settings from .env
load_dotenv()

# Shared state for concurrent execution
class ConcurState(TypedDict):
    question: str
    hr_answer: str
    biz_answer: str
    combined: str

# Set up Azure AD token provider using Azure CLI credential
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    azure_ad_token_provider=token_provider,
    temperature=0.5,
)

# Agent 1: HR perspective answer
def hr_node(state: ConcurState) -> ConcurState:
    q = state["question"]
    prompt = f"As an HR expert, answer the following question focusing on employee well-being:\n{q}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"hr_answer": response.content}

# Agent 2: Business perspective answer
def biz_node(state: ConcurState) -> ConcurState:
    q = state["question"]
    prompt = f"As a business operations expert, answer this question focusing on company productivity:\n{q}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"biz_answer": response.content}

# Aggregator: combine both answers
def aggregate_node(state: ConcurState) -> ConcurState:
    answer1 = state.get("hr_answer", "")
    answer2 = state.get("biz_answer", "")
    combined = f"HR perspective: {answer1}\n\nBusiness perspective: {answer2}"
    return {"combined": combined}

# Build the graph with parallel branches
graph_builder = StateGraph(ConcurState)
graph_builder.add_node("hr_agent", hr_node)
graph_builder.add_node("biz_agent", biz_node)
graph_builder.add_node("aggregate", aggregate_node)
# Start leads to both hr_agent and biz_agent in parallel
graph_builder.add_edge(START, "hr_agent")
graph_builder.add_edge(START, "biz_agent")
# Both branch outputs go into the aggregate node
graph_builder.add_edge("hr_agent", "aggregate")
graph_builder.add_edge("biz_agent", "aggregate")
graph_builder.add_edge("aggregate", END)
graph = graph_builder.compile()

# Run the graph on the example question
question_input = "What are the biggest benefits and challenges of remote work for a company?"
final_state = graph.invoke({"question": question_input})
print(final_state["combined"])
