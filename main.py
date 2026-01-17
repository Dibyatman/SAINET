from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode  <-- REMOVED
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage, 
    BaseMessage, 
    ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import asyncio
import json
from typing import TypedDict, Annotated, Sequence
import operator
from functools import partial

load_dotenv()

# --- 1. Define the Agent's State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retries: int

# --- 2. Define the Graph Nodes ---

def agent_node(state, agent_runnable):
    """
    Runs the main agent logic (model + tools)
    """
    print("---AGENT---")
    # Call the model. The 'agent_runnable' is our model bound to the tools
    response = agent_runnable.invoke(state['messages'])
    return {"messages": [response]}


async def tool_node(state, tool_map):
    """
    Runs the tool(s) called by the agent MANUALLY and ASYNCHRONOUSLY.
    """
    print("---TOOLS---")
    tool_calls = state['messages'][-1].tool_calls
    
    tool_messages = []
    for call in tool_calls:
        print(f"[TOOL CALL] {call['name']} -> {call['args']}")
        
        # Look up the tool in our map
        if call['name'] in tool_map:
            try:
                # Get the specific tool
                tool = tool_map[call['name']]
                
                # Use await .ainvoke() for async tools
                output = await tool.ainvoke(call['args']) 
                
                # Append a ToolMessage with the result
                tool_messages.append(
                    ToolMessage(content=str(output), tool_call_id=call['id'])
                )
            except Exception as e:
                print(f"Error calling tool {call['name']}: {e}")
                tool_messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=call['id'])
                )
        else:
            print(f"Error: Tool '{call['name']}' not found.")
            tool_messages.append(
                ToolMessage(content=f"Error: Tool '{call['name']}' not found", tool_call_id=call['id'])
            )
    
    print(f"[TOOL RESULTS] {tool_messages}")
    # Return the list of tool messages, which StateGraph will add to the state
    return {"messages": tool_messages}

def reflection_node(state, model):
    """
    The "critic" node. Checks the agent's final answer.
    """
    print("---REFLECT---")
    
    user_query = ""
    for msg in state['messages']:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
            
    agent_answer = state['messages'][-1].content
    
    # --- UPDATED: Critic prompt now understands tool errors vs. giving up ---
    reflection_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a critic. You must check the work of another agent.
You will be given the original user query and the agent's proposed final answer.
The agent's instructions are:
1.  For on-topic (agriculture) queries, it *must* use tools and provide a data-based answer OR a specific tool error.
2.  For off-topic queries or simple greetings, it *must* respond directly *without* using tools.

Your task: Check if the agent's final answer is acceptable.

* **Case 1: Query=On-Topic, Answer=Direct "I cannot help" / "I am an agriculture agent..."**
    * This is a **FAILURE**. The agent must use tools for on-topic queries and report the *specific* tool error if one occurred, not give up.
    * **Feedback:** This is an on-topic query. You must use the `APIKnowledgeBaseRetriever` tool to find the answer. Do not give up. If a tool fails, report the specific error or try a new plan. RETRY

* **Case 2: Query=Off-Topic/Greeting, Answer=Direct (No Tools)**
    * This is a **SUCCESS**. The agent correctly avoided using tools.
    * **Feedback:** ACCEPT

* **Case 3: Query=On-Topic, Answer=Data from tools OR a specific tool error message**
    * This is a **SUCCESS**. The agent correctly used its tools and provided a final answer (even if that answer is an error).
    * **Feedback:** ACCEPT
"""),
        ("user", "User Query: {query}\nAgent Answer: {answer}")
    ])
    
    critic_chain = reflection_prompt_template | model
    
    response = critic_chain.invoke({
        "query": user_query, 
        "answer": agent_answer 
    })
    
    print(f"[CRITIC] {response.content}")
    
    if "RETRY" in response.content.upper():
        return {
            "messages": [SystemMessage(content=f"Feedback: {response.content}")],
            "retries": state.get('retries', 0) + 1
        }
    else:
        return {"messages": [], "retries": state.get('retries', 0)}

# --- 3. Define the Graph's Conditional Edges ---

MAX_RETRIES = 3

def should_continue(state):
    """
    Decides where to go after the agent_node.
    """
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        # Agent called a tool -> go to tool_node
        return "call_tool"
    else:
        # Agent gave a final answer -> go to reflection_node to check it
        return "reflect"


def after_reflection(state):
    """
    Decides where to go after the reflection_node.
    """
    if state.get('retries', 0) >= MAX_RETRIES:
        print("---MAX RETRIES REACHED---")
        return END
    
    last_message = state['messages'][-1]
    if isinstance(last_message, SystemMessage) and "Feedback:" in last_message.content:
        return "retry_agent"
    else:
        return END

async def main():
    # Connect to MCP servers
    client = MultiServerMCPClient(
        {
            "APIKnowledgeBaseRetriever": {
                "url": "http://localhost:8000/mcp",  # retriever server
                "transport": "streamable_http",
            },
            "project-1": {
                "command": "python",
                "args": ["apiInvokeTool.py"],  # API executor
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}

    # Load model
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

    # --- 4. Create the Graph ---

    model_with_tools = model.bind_tools(tools)
    
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", partial(agent_node, agent_runnable=model_with_tools))
    workflow.add_node("tools", partial(tool_node, tool_map=tool_map))
    workflow.add_node("reflect", partial(reflection_node, model=model))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "call_tool": "tools",
            "reflect": "reflect"
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    workflow.add_conditional_edges(
        "reflect",
        after_reflection,
        {
            "retry_agent": "agent",
            "__end__": END
        }
    )

    app = workflow.compile()

    # --- UPDATED: This is the new, strict system message ---
    system_message = SystemMessage(content="""
You are a specialized reasoning and API-execution agent for agriculture data.
Your *only* purpose is to answer questions about agriculture, such as SPA, FPO, seeds, and crops, using the provided tools.
You must reason step-by-step to solve the user's query.

**Strict Rules:**

1.  **On-Topic Queries:** For any question related to agriculture data (e.g., "how many SPA in..."), you *must* use your tools. Follow this process:
    * **Step 1a:** Your *very first* action is to call the `APIKnowledgeBaseRetriever` tool. To get the correct API, you *must* pass the user's **latest, complete, and unmodified query** to this tool. Do not rephrase it.
    * **Step 1b:** Analyze the tool's output. If it mentions `preExecutionRequiredApi`, invoke that API *first* using the `project-1` tool.
    * **Step 1c:** Then, call the main API (also with `project-1`) using any required parameters.
    * **Step 1d:** After getting the API response, extract the relevant information to answer the user's question.

2.  **Off-Topic or Greetings:** If the user asks a question *not* related to agriculture (e.g., "what is the capital of France?") OR provides a simple greeting (e.g., "hi", "hello"), you *must* respond directly without using any tools.
    * For off-topic: Politely decline (e.g., "I am an agriculture agent and cannot help with that.")
    * For greetings: Respond with a simple greeting (e.g., "Hello! How can I assist with agriculture data today?")

3.  **Error Handling:** If an API call (like `api_invoke`) returns an error or a 400 status, you *must* re-evaluate your plan. Do not just give up. Report the *specific error* or re-examine the user's query and call `APIKnowledgeBaseRetriever` again to find a *different* API.

4.  **Schema:** When calling the `api_invoke` tool, **always follow this exact schema**:
    ```json
    {
      "method": "GET" | "POST",
      "url": "<api_endpoint>",
      "params": { ... },          // optional query params
      "body": {},                 // always an object, even if empty
      "headers": {}               // always an object, even if empty
    }
    ```
""")

    print("Welcome! I'm a self-correcting AI agent.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! 👋")
            break

        try:
            print("\n🔍 Executing reflection chain...\n")
            
            initial_state = {
                "messages": [system_message, HumanMessage(content=user_input)],
                "retries": 0
            }
            
            final_state = await app.ainvoke(initial_state)
            
            final_answer = "No final answer found."
            if final_state and final_state.get("messages"):
                for msg in reversed(final_state["messages"]):
                    if msg.type == 'ai' and not msg.tool_calls:
                        final_answer = msg.content
                        break
            
            print(f"\n[AGENT FINAL ANSWER] {final_answer}\n")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    