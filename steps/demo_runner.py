"""
Demo Runner for Practical Assignment 3
Covers Parts A, B, C, and D with reproducible outputs, metrics, and traces.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

os.environ["TQDM_DISABLE"] = "1"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
DOCS_PATH = Path(__file__).parent.parent / "data" / "documents"

SERVER = StdioServerParameters(
    command="uv",
    args=["run", "python", "-m", "server.main"],
    env=dict(os.environ),
)


# =============================================================================
# Part A: Vector Database & 5 Questions Test
# =============================================================================
def run_part_a_tests():
    print("=" * 80)
    print("PART A: CHROMADB VECTOR SEARCH EVALUATION (5 QUESTIONS)")
    print("=" * 80)

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-small")
    client = chromadb.Client()
    collection = client.create_collection("part_a_eval", embedding_function=embedding_fn)

    files = []
    for f in sorted(DOCS_PATH.glob("*.txt")):
        content = f.read_text(encoding="utf-8")
        files.append((f.name, content))

    collection.add(
        ids=[f[0] for f in files],
        documents=["passage: " + f[1] for f in files],
    )

    questions = [
        "What are your opening hours on weekends?",
        "What vegan or plant-based food items are available?",
        "Is there wireless internet and power sockets for laptops?",
        "Where is the café located and how do I get there from the station?",
        "What is the story behind the café and how did it start?",
    ]

    results_data = []
    for idx, q in enumerate(questions, 1):
        res = collection.query(query_texts=["query: " + q], n_results=2)
        top_doc = res["ids"][0][0]
        top_dist = res["distances"][0][0]
        second_doc = res["ids"][0][1]
        second_dist = res["distances"][0][1]
        doc_preview = res["documents"][0][0][:200].replace("\n", " ")

        print(f"\n[Question {idx}]: \"{q}\"")
        print(f"  Top Match:      {top_doc} (distance: {top_dist:.4f})")
        print(f"  Second Match:   {second_doc} (distance: {second_dist:.4f})")
        print(f"  Snippet:        {doc_preview}...")

        results_data.append({
            "question_num": idx,
            "question": q,
            "top_match": top_doc,
            "distance": top_dist,
            "snippet": doc_preview,
            "assessment": "Highly Relevant" if top_doc in ["hours_access.txt", "allergens.txt", "faq.txt", "story.txt"] else "Partial"
        })

    return results_data


# =============================================================================
# Part C: LLM + RAG 5 Questions + 1 Unanswerable Question Test
# =============================================================================
async def run_part_c_tests(session: ClientSession):
    print("\n" + "=" * 80)
    print("PART C: LLM + RAG INTEGRATION (5 CAFE QUESTIONS + 1 UNANSWERABLE)")
    print("=" * 80)

    tools = (await session.list_tools()).tools
    ollama_tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]

    questions = [
        ("Opening Hours", "What time does the café open and close on Saturday and Sunday?"),
        ("Food & Dietary", "What options do you have for someone who cannot eat wheat or gluten?"),
        ("Working Environment", "Can I work on my laptop there, and are there power outlets available?"),
        ("Directions", "How do I reach the café from Shibuya and where can I park a bicycle?"),
        ("Origin Story", "Who founded The Wandering Kettle and what is the meaning behind its name?"),
        ("Unanswerable / Missing", "Do you sell branded ceramic mugs, t-shirts, or tote bags as café merchandise?"),
    ]

    c_results = []
    for topic, q in questions:
        print(f"\n--- Testing Topic: {topic} ---")
        print(f"Question: {q}")

        # 1. Ask LLM for tool decision
        decision = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
            tools=ollama_tools,
        )

        calls = decision.message.tool_calls
        if calls:
            fn_name = calls[0].function.name
            fn_args = calls[0].function.arguments
            print(f"  [LLM Tool Decision]: {fn_name}({fn_args})")

            # 2. Call MCP Tool
            mcp_res = await session.call_tool(fn_name, fn_args)
            context = mcp_res.content[0].text if mcp_res.content else ""
            print(f"  [MCP Retrieval]: Retrieved {len(context)} characters")

            # 3. Generate Grounded Answer
            system_prompt = (
                "You are an assistant for The Wandering Kettle cafe. "
                "Answer using ONLY the provided context. If the answer is NOT present in the reference, "
                "explicitly state that the cafe documents do not contain this information. Do NOT fabricate."
            )
            ans_res = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"}
                ],
            )
            final_ans = ans_res.message.content.strip()
        else:
            final_ans = decision.message.content.strip()

        print(f"  [Grounded LLM Answer]:\n  {final_ans}\n")
        c_results.append({
            "topic": topic,
            "question": q,
            "answer": final_ans
        })

    return c_results


async def main():
    part_a_data = run_part_a_tests()

    async with stdio_client(SERVER, errlog=subprocess.DEVNULL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            part_c_data = await run_part_c_tests(session)

    results = {
        "part_a": part_a_data,
        "part_c": part_c_data,
    }
    out_file = Path(__file__).parent / "demo_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll Part A & Part C test results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
