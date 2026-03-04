# Agentic Workflow Pattern Examples

This repository contains **10 Python examples** demonstrating five key multi-agent orchestration patterns using two frameworks:

- **Azure AI Agent Framework SDK** (part of Azure AI Foundry)
- **LangChain's LangGraph** (state graph orchestration in LangChain)

Each pattern has two examples (one for each framework). The patterns covered are:

1. **Sequential** – Agents executed in order (pipeline).
2. **Concurrent** – Agents executed in parallel with result aggregation.
3. **Group Chat** – Multiple agents taking turns in a moderated conversation.
4. **Handoff** – Routing a conversation to different agents based on context.
5. **Magentic** – A manager (planner) agent dynamically breaking down a complex goal into sub-tasks and coordinating worker agents.

## Getting Started

### Requirements

- Python 3.8+
- An Azure OpenAI resource (for Agent Framework examples) with appropriate models deployed (e.g., GPT-3.5-Turbo or GPT-4).
- OpenAI API key (for LangChain examples, or configure LangChain to use Azure OpenAI).
- The following Python packages (see `requirements.txt`):
  - `agent-framework` (Microsoft Agent Framework SDK for Python)
  - `azure-identity` (for Azure AD authentication in Agent Framework)
  - `python-dotenv` (to load Azure OpenAI settings from a .env file)
  - `langchain` (for LangGraph)
  - `openai` (OpenAI Python client, used by LangChain)
  
Install dependencies with:
```bash
pip install -r requirements.txt
```

### Configuration

For the Azure AI Agent Framework examples, set up access to your Azure OpenAI resource. You can either:
- **Use Azure AD (development):** Ensure you have run `az login` with an account that has access to the Azure OpenAI resource. The examples default to using `AzureCliCredential()`.
- **Or use API key and endpoint:** Create a `.env` file in the project root (or set environment variables) with the following:
  ```
  AZURE_OPENAI_ENDPOINT=<your Azure OpenAI endpoint>
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<your Azure OpenAI deployment name>
  AZURE_OPENAI_API_KEY=<your Azure OpenAI API key>
  AZURE_OPENAI_API_VERSION=<optional, e.g., 2023-03-15-preview>
  ```
  The Agent Framework will load these values automatically via `load_dotenv()`.

For the LangChain examples, if using OpenAI's API:
- Set the environment variable `OPENAI_API_KEY` to your OpenAI API key.

To use Azure OpenAI with LangChain instead, configure the OpenAI SDK accordingly (see LangChain documentation for Azure OpenAI usage).

### Running the Examples

Each example is a standalone Python script. Run them individually. For example:
```bash
python sequential_agent_framework.py
```
to run the sequential pattern with the Azure Agent Framework, or:
```bash
python sequential_langgraph.py
```
to run the equivalent using LangChain's LangGraph.

Each script contains comments explaining the workflow logic.

**Note:** Running these examples will invoke language model API calls. Ensure you have the necessary credentials set and be aware of usage costs on your Azure/OpenAI account.

## Example Overview

| **Pattern**    | **Framework**            | **Description**                                                |
| -------------- | ----------------------- | -------------------------------------------------------------- |
| Sequential     | Agent Framework SDK     | Summarize a report then provide an insight (two-step pipeline) |
| Sequential     | LangChain LangGraph     | Summarize a report then provide an insight (two-step pipeline) |
| Concurrent     | Agent Framework SDK     | Parallel answers from HR and Business agents to a question     |
| Concurrent     | LangChain LangGraph     | Parallel answers from HR and Business agents to a question     |
| Group Chat     | Agent Framework SDK     | Moderated debate with alternating turns (pro vs. con)          |
| Group Chat     | LangChain LangGraph     | Moderated debate with alternating turns (pro vs. con)          |
| Handoff        | Agent Framework SDK     | Support chatbot handing off to a refund specialist             |
| Handoff        | LangChain LangGraph     | Routing to refund or general support agent based on query      |
| Magentic       | Agent Framework SDK     | AI planner agent decomposing a goal and coordinating workers   |
| Magentic       | LangChain LangGraph     | Dynamic workflow that plans tasks and executes them iteratively |

