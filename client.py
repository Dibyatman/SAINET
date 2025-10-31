from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import asyncio
import json


load_dotenv()

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

    # Get available tools
    tools = await client.get_tools()

    # Load model
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

    # Create the agent (no messages argument here)
    agent = create_react_agent(model, tools)

    # Define your reasoning/system instruction
    system_message = SystemMessage(content="""
You are a reasoning and API-execution agent.

When a user asks a question, follow this reasoning process:

1. Use the `APIKnowledgeBaseRetriever` tool first to find which API can answer the query.
2. If the retrieved API mentions a `preExecutionRequiredApi`, invoke that first using the `project-1` tool to get any required parameters.
3. Then call the main API (also with the `project-1` tool) using those parameters mostly place holders like "stateLGDCode", "cropCode", "varietyCode" ..

When calling the `api_invoke` tool, **always follow this exact schema**:
```json
{
  "method": "GET" | "POST",
  "url": "<api_endpoint>",
  "params": { ... },          // optional query params
  "body": {},                 // always an object, even if empty
  "headers": {}               // always an object, even if empty
}
 4. After getting the API response, extract the relevant information to answer the user's question.
5. If at any point an error occurs during API invocation, report the error message back to the user.
Always think step-by-step and ensure to call APIs in the correct order based on dependencies.                                  
Provide clear and concise answers based on the API data retrieved.""")


    print("Welcome! I'm an AI agent with live API access.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! 👋")
            break


        def safe_json_dump(obj):
            """Safely convert any object to printable JSON."""
            try:
                return json.dumps(obj, indent=2, ensure_ascii=False)
            except TypeError:
                if isinstance(obj, list):
                    return json.dumps(
                        [str(o) for o in obj], indent=2, ensure_ascii=False
                    )
                return json.dumps(str(obj), indent=2, ensure_ascii=False)


        try:
            print("\n🔍 Executing reasoning chain...\n")
            async for event in agent.astream_events(
                {"messages": [system_message, HumanMessage(content=user_input)]},
                version="v1"
            ):
                kind = event.get("event", "")
                data = event.get("data", {})  # ✅ Correct event data key

                # TOOL CALL START
                if kind == "on_tool_start":
                    tool_name = data.get("name")
                    tool_input = data.get("input", {})
                    print(str(f"[TOOL CALL] {tool_name} -> {safe_json_dump(tool_input)}"))

                # TOOL RESULT END
                elif kind == "on_tool_end":
                    tool_output = data.get("output", {})
                    print(str(f"[TOOL RESULT] -> {safe_json_dump(tool_output)}"))

                # FINAL MODEL OUTPUT
                elif kind == "on_chat_model_end":
                    output_data = data.get("output", {})
                    content = None

                    # 🧠 Try multiple possible formats depending on model backend
                    if isinstance(output_data, dict):
                        # 1️⃣ Newer LangChain chat output
                        if "messages" in output_data:
                            messages = output_data["messages"]
                            if messages and isinstance(messages[-1], dict):
                                content = messages[-1].get("content")
                        # 2️⃣ Older / Groq / LLM format
                        elif "generations" in output_data:
                            gens = output_data["generations"]
                            if gens and isinstance(gens[0], list) and len(gens[0]) > 0:
                                gen = gens[0][0]
                                content = gen.get("text") or gen.get("content")
                        # 3️⃣ Direct content field (some LangGraph builds)
                        elif "content" in output_data:
                            content = output_data["content"]

                    if content:
                        print(f"\n[AGENT FINAL ANSWER] {content}\n")
                    else:
                        print("\n[AGENT FINAL ANSWER] ⚠️ No final message content found.\n")


        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
