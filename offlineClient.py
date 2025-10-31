from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Dict, Any, Optional, Sequence, Union
from pydantic import Field
import json
import asyncio

load_dotenv()

# --- TinyLlamaChatWrapper Class (No changes needed here) ---
class TinyLlamaChatWrapper(BaseChatModel):
    pipeline: Any = Field(default=None, exclude=True)
    bound_tools: Optional[List[BaseTool]] = Field(default=None, exclude=True)

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.bound_tools = None

    @property
    def _llm_type(self) -> str:
        return "tinyllama-chat"

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "TinyLlamaChatWrapper":
        """Bind tools to the chat model."""
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        # Convert messages to prompt format TinyLlama expects
        prompt = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt += f"### Human: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                prompt += f"### System Instruction: {msg.content}\n" 
            else:
                prompt += f"### Assistant: {msg.content}\n"
        prompt += "### Assistant:"
        
        # Generate response
        result = self.pipeline(prompt, **kwargs)
        # Attempt to get only the new assistant response
        response = result[0]['generated_text'].split("### Assistant:")[-1].strip()
        
        # Return as ChatResult with generations
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        return self._generate(messages, **kwargs)
# -----------------------------------------------------------

# 💡 Helper function for safe JSON dumping (kept in case you want to re-add debug)
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

    # Load the local TinyLlama model
    model_id = "./tinyllama-local"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.4,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15
    )
    
    llm = TinyLlamaChatWrapper(pipeline=pipe)
    tools = await client.get_tools()
    
    # Bind tools before creating agent
    llm_with_tools = llm.bind_tools(tools)
    agent = create_react_agent(llm_with_tools, tools)

    # 💡 System instruction
    system_message = SystemMessage(content="""
You are an expert reasoning and API-execution agent. Your primary language is English.

< ALERT > When a user asks a question, follow this reasoning process:

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
Provide clear and concise answers based on the API data retrieved.
""")


# ----------------- user input -----------------
    print("Welcome! I'm an AI agent with access to tools.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        # 1. Get user input
        user_input = input("You: ")
        
        # 2. Check for exit command
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! 👋")
            break
        
        # 3. Invoke the agent with the user's message
        try:
            print("\n🔍 Executing chain...")
            # Use ainvoke to get the final result after all steps are complete
            agent_response = await agent.ainvoke(
                # Pass both the system message and the user's message
                {"messages": [system_message, HumanMessage(content=user_input)]}
            )
            
            # 4. Print the final response from the agent
            # The last message in the 'messages' list is the final answer.
            final_message = agent_response['messages'][-1]

            if isinstance(final_message, AIMessage):
                print("Agent:", final_message.content)
            else:
                 # In case the final message isn't an AIMessage (e.g., if it's a ToolMessage after an error)
                 print("Agent:", final_message.content or "Error: The agent did not return a final AI message.")

            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())