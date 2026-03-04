from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

# Load Azure OpenAI settings from .env
load_dotenv()

# State for the group chat simulation
class ChatState(TypedDict):
    history: List[str]      # conversation history (strings like 'Speaker: message')
    turn: int
    next_speaker: str       # either "ProAgent" or "ConAgent"
    topic: str
    done: bool

# Set up Azure AD token provider using Azure CLI credential
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    azure_ad_token_provider=token_provider,
    temperature=0.9,
)

# Agent 1 (ProAgent) node: adds a pro argument to history
def pro_agent_node(state: ChatState) -> ChatState:
    conversation = "\n".join(state["history"])
    system_msg = SystemMessage(content="You are ProAgent, who argues in favor of the topic. Keep it concise (2-3 sentences).")
    user_msg = HumanMessage(content=f"{conversation}\nProAgent:")
    response = llm.invoke([system_msg, user_msg])
    new_history = state["history"] + [f"ProAgent: {response.content.strip()}"]
    return {"history": new_history, "next_speaker": "ConAgent", "turn": state["turn"] + 1}

# Agent 2 (ConAgent) node: adds a con argument to history
def con_agent_node(state: ChatState) -> ChatState:
    conversation = "\n".join(state["history"])
    system_msg = SystemMessage(content="You are ConAgent, who argues against the topic. Keep it concise (2-3 sentences).")
    user_msg = HumanMessage(content=f"{conversation}\nConAgent:")
    response = llm.invoke([system_msg, user_msg])
    new_history = state["history"] + [f"ConAgent: {response.content.strip()}"]
    return {"history": new_history, "next_speaker": "ProAgent", "turn": state["turn"] + 1}

# Router function: decides next node based on state
def route_next(state: ChatState) -> str:
    if state.get("turn", 0) >= 4:
        return "end"
    if state.get("next_speaker") == "ConAgent":
        return "con_turn"
    return "pro_turn"

# Build the graph with conditional turn-taking
graph_builder = StateGraph(ChatState)
graph_builder.add_node("pro_turn", pro_agent_node)
graph_builder.add_node("con_turn", con_agent_node)

# Start always goes to pro_turn first
graph_builder.add_edge(START, "pro_turn")

# After pro_turn, route to con or end
graph_builder.add_conditional_edges("pro_turn", route_next, {"con_turn": "con_turn", "pro_turn": "pro_turn", "end": END})

# After con_turn, route to pro or end
graph_builder.add_conditional_edges("con_turn", route_next, {"pro_turn": "pro_turn", "con_turn": "con_turn", "end": END})

graph = graph_builder.compile()

# Initial state with the topic and an empty history (user sets topic)
initial_state = {
    "topic": "Should AI be strictly regulated by the government?",
    "history": ["User: Should AI be strictly regulated by the government?"],
    "turn": 0,
    "next_speaker": "ProAgent",
    "done": False
}
final_state = graph.invoke(initial_state)
print("Conversation Transcript:")
for line in final_state["history"]:
    print(line)
