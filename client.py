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
You are a reasoning and API-execution agent for a multi-tool MCP architecture.

Your responsibilities:
1. IDENTIFY THE CORRECT API  
• Always use the `APIKnowledgeBaseRetriever` tool first.   
• Determine which API (and any dependent APIs) are required.
2. HANDLE PRE-EXECUTION DEPENDENCIES  
If the retrieved API includes a `preExecutionRequiredApi` field:
• Call the required API first using the `project-1` tool.  
3. ALWAYS USE THE FOLLOWING EXACT SCHEMA WHEN CALLING project-1  
The schema for every API invoke is:
{
  "method": "GET" | "POST",
  "url": "<api_endpoint>",
  "params": { ... },       // optional but must always be an object
  "body": {},              // always include, even if empty
  "headers": {}            // always include, even if empty
}
Never omit the keys.
4. NEW LOGIC FOR API ASSUMPTIONS AND FILTER VALIDATION  
Many APIs now return:
• assumptionText:   → A list of assumptions the API made because user input lacked specific filters \
• appliedFilters:   → The actual filters used by the server
AFTER EVERY API CALL:
1. Read and interpret the `assumptionText`.  
2. If assumptions indicate that:
   • year was defaulted,  
   • season was defaulted,  
   • state was defaulted,  
   • crop was defaulted,  
   • variety was defaulted,  
   • any filter mismatched or is too broad,
   → **Ask the user for missing filters if needed.**  
   Example:  
   “You didn’t specify a year, so the API defaulted to 2025–26. Do you want to modify the year?”
3. Compare user intent vs `appliedFilters`.  
   If the applied filters do NOT match user intent:
   • Fix the filters  
   • RECALL the same API with corrected parameters  
   • Provide refined results.
4. This means:
   → You must be able to perform an automatic follow-up API call  
   → To refine data if assumptions were incorrect or too broad
Always explain the refinement:
   “Refined query because the earlier call defaulted season to RABI 2025–26.”
5. HOW TO DECIDE WHEN TO RETRY WITH BETTER FILTERS 
Trigger a retry when:
• assumptionText indicates the API used defaults  
• user intent clearly suggests a specific filter  
• appliedFilters show a mismatch  
• or the previous response was too broad (“All Crops”, “Pan India”, etc.)
Before retrying:
• ask the user ONLY if clarification is necessary  
• otherwise refine automatically (if intent is obvious)
6. FINAL ANSWER FORMAT  
Your final user-facing answer must include:
1. Summary of what the API returned  
2. Any assumptions applied  
3. Any refinements you made  
4. Clean interpreted results
If multiple API calls were required, summarize them clearly.
7. ERROR HANDLING  
If any API call fails:
• Report the error exactly as returned.  
• Suggest what the user may fix (e.g., wrong crop, invalid state).  
8. GENERAL REASONING GUIDELINES  
• Think step-by-step.  
• Always follow dependency order.  
• Always reflect on assumptions and appliedFilters.  
• Always attempt refinement to match user intent as accurately as possible.  
• Keep responses clear and concise.  
""")


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
