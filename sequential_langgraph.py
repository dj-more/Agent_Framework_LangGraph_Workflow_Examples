from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

# Load Azure OpenAI settings from .env
load_dotenv()

# Set up Azure AD token provider using Azure CLI credential
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

# Define state for sequential workflow
class SeqState(TypedDict):
    text: str       # the text to summarize
    summary: str
    insight: str

# Initialize Azure OpenAI chat model using gpt-4o deployment
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    azure_ad_token_provider=token_provider,
    temperature=0.7,
)

# Node 1: Summarize the text into key bullet points
def summarize_node(state: SeqState) -> SeqState:
    text = state["text"]
    prompt = f"Summarize the following text into 2-3 bullet points:\n{text}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

# Node 2: Provide an insight based on the summary
def insight_node(state: SeqState) -> SeqState:
    summary = state["summary"]
    prompt = f"Read the summary below and provide one key insight or conclusion based on it:\n{summary}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"insight": response.content}

# Build the state graph for sequential execution
graph_builder = StateGraph(SeqState)
graph_builder.add_node("summarize", summarize_node)
graph_builder.add_node("insight", insight_node)
graph_builder.add_edge(START, "summarize")  # first the summarization step
graph_builder.add_edge("summarize", "insight")  # then the insight step
graph_builder.add_edge("insight", END)
graph = graph_builder.compile()

# Example text to summarize
text_to_process = (
    "AcmeCorp's Q4 report shows revenue grew by 20% to $5M, but net profits declined due to higher expenses. "
    "The company plans to expand to new markets next year."
)
# Run the graph with the example text
output_state = graph.invoke({"text": text_to_process})
print("Summary:\n", output_state["summary"])
print("Insight:\n", output_state["insight"])
