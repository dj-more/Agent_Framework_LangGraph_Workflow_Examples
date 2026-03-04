from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

# Load Azure OpenAI settings from .env
load_dotenv()

# Define state for routing
class SupportState(TypedDict):
    query: str
    category: str
    answer: str

# Set up Azure AD token provider using Azure CLI credential
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    azure_ad_token_provider=token_provider,
    temperature=0,
)

# Classifier node: determine query type
def classify_node(state: SupportState) -> SupportState:
    q = state["query"].lower()
    category = "refund" if ("refund" in q or "return" in q) else "general"
    return {"category": category}

# Router function for conditional edges
def route_by_category(state: SupportState) -> str:
    return "refund_handler" if state.get("category") == "refund" else "general_handler"

# General support agent node
def general_node(state: SupportState) -> SupportState:
    q = state["query"]
    prompt = f"You are a customer support assistant. Answer the question concisely:\n{q}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": f"SupportAgent: {response.content}"}

# Refund specialist agent node
def refund_node(state: SupportState) -> SupportState:
    q = state["query"]
    prompt = f"You are a refund specialist. The customer asks:\n{q}\nProvide a helpful response."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": f"RefundAgent: {response.content}"}

# Build the graph with branching
graph_builder = StateGraph(SupportState)
graph_builder.add_node("classifier", classify_node)
graph_builder.add_node("general_handler", general_node)
graph_builder.add_node("refund_handler", refund_node)
graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges("classifier", route_by_category, {"refund_handler": "refund_handler", "general_handler": "general_handler"})
graph_builder.add_edge("general_handler", END)
graph_builder.add_edge("refund_handler", END)
graph = graph_builder.compile()

# Example queries
queries = [
    "How do I change my account password?",
    "I want to return my product and get a refund."
]
for q in queries:
    output_state = graph.invoke({"query": q})
    print(output_state["answer"])
    print()
