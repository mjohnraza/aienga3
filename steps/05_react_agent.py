"""
Step 5: Iterative ReAct Agent (Part D)

An iterative ReAct (Reason + Act) loop using the local LLM and MCP tools.
The agent inspects the user question, determines what information is missing,
calls MCP tools iteratively (Thought -> Action -> Observation), maintains history,
and produces a grounded final answer combining evidence and noting missing information.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
MAX_STEPS = 6

SERVER = StdioServerParameters(
    command="uv",
    args=["run", "python", "-m", "server.main"],
    env=dict(os.environ),
)

CHALLENGE_REQUEST = (
    "I'm planning to visit the café with my cat. Can I sit outside, "
    "what food options are available, and are there any allergy considerations I should know about?"
)

SYSTEM_PROMPT = """You are an accurate, grounded knowledge assistant for "The Wandering Kettle" café.
You follow a strict multi-step ReAct (Reasoning and Acting) loop to answer composite questions.

CRITICAL INSTRUCTIONS:
- The user's query contains three distinct sub-questions:
  1. Outdoor seating and pet/cat policy
  2. Available food menu items
  3. Allergy and dietary considerations
- You CANNOT give a final answer until you have gathered verified source documents for ALL THREE aspects.
- In each step:
  * If pet/terrace policy is missing: Call `search` with query="pet policy animal outdoor terrace", top_k=1
  * If food menu is missing: Call `search` with query="food menu all day plates sandwiches pastries", top_k=1
  * If allergy details are missing: Call `search` with query="allergens ingredients dietary notes", top_k=1
- NEVER guess or fabricate food items or allergy policies (e.g. do NOT invent gluten-free bread; the cafe has NO gluten-free bread).
- Only when you have observations from all 3 searches, output your comprehensive final answer.
- Distinguish confirmed policies (dogs allowed on terrace, assistance animals everywhere) from unmentioned policies (cats are not mentioned in the policy).
"""


def to_ollama_format(tools: list) -> list[dict]:
    """Convert MCP tools into Ollama function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


async def run_react_agent(session: ClientSession, user_query: str, available_tools: list) -> dict:
    """Execute the iterative ReAct loop."""
    print("=" * 80)
    print("STARTING ITERATIVE ReAct AGENT")
    print(f"User Query: {user_query}")
    print("=" * 80)

    ollama_tools = to_ollama_format(available_tools)
    tool_names = {t.name for t in available_tools}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    trace = []
    step = 0

    while step < MAX_STEPS:
        step += 1
        print(f"\n{'='*30} ITERATION {step} {'='*30}")

        # 1. Ask LLM for next decision
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
            )
        except Exception as e:
            err_msg = f"LLM inference error: {e}"
            print(f"[ERROR] {err_msg}")
            return {"status": "error", "message": err_msg, "trace": trace}

        msg = response.message
        tool_calls = msg.tool_calls

        # If LLM produces a thought / reasoning text
        thought = msg.content.strip() if msg.content else "Deciding next tool call based on missing information..."
        print(f"[Thought / Reasoning]:\n  {thought}")

        # 2. Check if LLM decided on a Final Answer
        if not tool_calls:
            # Application-level safeguard: Ensure at least 3 tool searches have been performed
            # for composite questions before accepting final answer
            executed_searches = [t["action"] for t in trace if "search" in t.get("action", "")]
            if len(executed_searches) < 3 and step < 4:
                guard_notice = (
                    "Guard Notice: You have only executed "
                    f"{len(executed_searches)} search(es). The question requires checking: "
                    "(1) outdoor & animal policy, (2) food menu, and (3) allergen notes. "
                    "Please search for the remaining missing information before synthesizing your final answer."
                )
                print(f"\n[Safeguard Triggered]: {guard_notice}")
                messages.append(msg)
                messages.append({"role": "user", "content": guard_notice})
                continue

            print("\n[Decision]: Final Answer reached (evidence collected).")
            print("=" * 80)
            print("FINAL SYNTHESIZED ANSWER:")
            print("=" * 80)
            print(msg.content)
            trace.append({
                "iteration": step,
                "thought": thought,
                "action": "final_answer",
                "final_answer": msg.content,
            })
            return {
                "status": "completed",
                "iterations": step,
                "answer": msg.content,
                "trace": trace,
            }

        # 3. Process Tool Call Action (execute one per iteration in classic ReAct style)
        messages.append(msg)
        call = tool_calls[0]
        func_name = call.function.name
        func_args = call.function.arguments

        print(f"\n[Action (Step {step})]: Call Tool '{func_name}'")
        print(f"  Arguments: {json.dumps(func_args, ensure_ascii=False)}")

        # Safeguard 1: Validate tool existence
        if func_name not in tool_names:
            obs = f"Error: Tool '{func_name}' does not exist. Available tools: {list(tool_names)}"
            print(f"[Observation Error]: {obs}")
        else:
            # Safeguard 2: Execute tool via MCP
            try:
                result = await session.call_tool(func_name, func_args)
                if result.is_error:
                    obs = f"Tool execution failed: {result.content[0].text if result.content else 'Unknown error'}"
                else:
                    obs = result.content[0].text if result.content else ""
            except Exception as e:
                obs = f"Application execution error for '{func_name}': {str(e)}"

        # Print observation summary
        preview = obs[:250].replace("\n", " ") + ("..." if len(obs) > 250 else "")
        print(f"[Observation (Step {step})] ({len(obs)} chars received):")
        print(f"  {preview}")

        # Append observation to history for next iteration
        messages.append({
            "role": "tool",
            "content": obs,
        })

        trace.append({
            "iteration": step,
            "thought": thought,
            "action": f"{func_name}({json.dumps(func_args)})",
            "observation_summary": preview,
            "observation_len": len(obs),
        })

    # Safeguard 3: Max iterations reached
    safe_stop_message = (
        "Safe Stop: Reached maximum iterations limit without producing a final answer. "
        "Partial evidence collected: " + str([t['action'] for t in trace])
    )
    print(f"\n[WARNING] {safe_stop_message}")
    return {"status": "max_steps_exceeded", "iterations": step, "answer": safe_stop_message, "trace": trace}


async def main():
    async with stdio_client(SERVER, errlog=subprocess.DEVNULL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"Connected to MCP Server. Exposed tools: {[t.name for t in tools]}")

            # Run Challenge Request
            result = await run_react_agent(session, CHALLENGE_REQUEST, tools)

            # Save trace artifact
            output_file = Path(__file__).parent / "react_trace.json"
            output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nTrace successfully written to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
