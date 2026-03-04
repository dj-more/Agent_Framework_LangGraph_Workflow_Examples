from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

# Load Azure OpenAI settings from .env
load_dotenv()

# State for dynamic planning workflow
class PlanState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    done: bool

# Set up Azure AD token provider using Azure CLI credential
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    azure_ad_token_provider=token_provider,
    temperature=0.7,
)

# Planner node: decide tasks or finalize answer
def planner_node(state: PlanState) -> PlanState:
    tasks = state.get("tasks", [])
    results = state.get("results", [])
    # If we have results and no remaining tasks, compile final answer
    if results and not tasks:
        combined_results = "\n".join(results)
        prompt = f"Based on the following results, provide a concise conclusion for the goal:\n{combined_results}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"done": True, "results": results + [f"Final Answer: {response.content.strip()}"]}
    # If no tasks and no results, create initial plan
    if not tasks and not results:
        return {"tasks": ["Find the latest GDP of France", "Analyze how France's GDP affects the European economy"], "done": False}
    return {}

# Worker node: execute the next task
def worker_node(state: PlanState) -> PlanState:
    if state["tasks"]:
        task = state["tasks"][0]
        remaining = state["tasks"][1:]
        # Use LLM to perform the task
        prompt = f"Complete the following task concisely:\n{task}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"tasks": remaining, "results": state.get("results", []) + [f"{task}: {response.content.strip()}"]}
    return {}

# Router: decide whether to continue or finish
def route_planner(state: PlanState) -> str:
    if state.get("done"):
        return "end"
    if state.get("tasks"):
        return "worker"
    return "end"

# Build the loop graph (planner -> worker -> planner)
graph_builder = StateGraph(PlanState)
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("worker", worker_node)
graph_builder.add_edge(START, "planner")
graph_builder.add_conditional_edges("planner", route_planner, {"worker": "worker", "end": END})
graph_builder.add_edge("worker", "planner")
graph = graph_builder.compile()

# Run the dynamic planning graph
initial_state = {"goal": "Find the latest GDP of France and analyze what it means for the European economy.", "tasks": [], "results": [], "done": False}
final_state = graph.invoke(initial_state)
print("Task Results and Final Answer:")
for res in final_state["results"]:
    print(f"\n{res}")
