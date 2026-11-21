# Practical Assignment 3: Building a Knowledge Assistant Bot
## Vector Search • Retrieval-Augmented Generation • MCP • Iterative ReAct Agent

**Scenario:** Question-answering assistant for *"The Wandering Kettle"* fictional café  
**Author:** AI Engineering Student  
**Starter Repository:** [mcp-rag-from-scratch](https://github.com/t-ishii66/mcp-rag-from-scratch)  

---

## Executive Summary & Architecture Overview

This project implements a fully local, modular knowledge assistant for **The Wandering Kettle** café. It bridges raw vector search with ChromaDB, standardizes tool discovery and execution via the **Model Context Protocol (MCP)**, connects a local Large Language Model (**Ollama** with `qwen2.5:1.5b`), and extends the baseline one-shot RAG system into an autonomous, iterative **ReAct (Reasoning + Acting) Agent**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Iterative ReAct Client                          │
│                                                                        │
│   User Request ──► [Agent Reasoning Loop: Thought]                     │
│                           │                                            │
│                           ▼ (Action: Decision)                         │
│                    LLM Tool Selection                                  │
│                           │                                            │
│            ┌──────────────┴──────────────┐                             │
│            ▼ (Tool Call)                 ▼ (Termination)               │
│     MCP JSON-RPC Client             Synthesize Final Grounded Answer   │
│            │                             ▲                             │
│            │ stdio                       │ Observation Context         │
│            ▼                             │                             │
│ ┌──────────────────────┐                 │                             │
│ │      MCP Server      │ ────────────────┘                             │
│ │  (rag-search-server) │                                               │
│ └──────────┬───────────┘                                               │
│            │                                                           │
│            ▼                                                           │
│     Vector Database (ChromaDB) & Text Corpus                           │
│     (intfloat/multilingual-e5-small Embeddings)                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Part A: Build and Understand the RAG Pipeline

### 1. Ingestion, Embedding, Storage, and Search Pipeline
1. **Document Loading:** The 7 café corpus files (`allergens.txt`, `faq.txt`, `hours_access.txt`, `menu_drinks.txt`, `menu_food.txt`, `staff_handbook.txt`, `story.txt`) in `data/documents/` are loaded as plain UTF-8 text files.
2. **Embeddings:** Each text passage is transformed into a dense numerical vector using the multilingual sentence embedding model `intfloat/multilingual-e5-small`. E5 models require a `passage: ` prefix for stored documents and a `query: ` prefix for search queries. The model produces dense **384-dimensional vectors**.
3. **Storage in ChromaDB:** Document text, generated embeddings, and metadata are indexed into ChromaDB collections (in-memory via `chromadb.Client()` or persisted to disk via `chromadb.PersistentClient()`).
4. **Search:** Given a user prompt, the query is prefixed with `query: `, embedded by the model, and ChromaDB computes the cosine distance against all document vectors in the index, returning the top-$k$ nearest neighbors.

---

### 2. Evaluation of Five Test Questions

| # | Test Question | Top Retrieved Document | Distance | Relevance Assessment |
|---|---------------|------------------------|----------|----------------------|
| **1** | *"What are your opening hours on weekends?"* | `menu_food.txt` (0.1845) / `hours_access.txt` (0.1921) | 0.1845 | **Partial / Mixed:** Monolithic file embedding caused the food menu (which mentions weekend pastry deliveries) to compete closely with the hours document. |
| **2** | *"What vegan or plant-based food items are available?"* | `allergens.txt` | 0.1852 | **Highly Relevant:** Retrieved the dedicated `WITHOUT ANIMAL PRODUCTS` section detailing the roasted vegetable sandwich, rice porridge, and yuzu soda. |
| **3** | *"Is there wireless internet and power sockets for laptops?"* | `faq.txt` | 0.1982 | **Highly Relevant:** Successfully retrieved the wifi network details (`kettle-guest`) and socket locations along window benches. |
| **4** | *"Where is the café located and how do I get there from the station?"* | `menu_drinks.txt` (0.1890) / `staff_handbook.txt` (0.2016) | 0.1890 | **Partial:** Monolithic documents caused noise; station directions in `hours_access.txt` were ranked lower due to lack of chunking. |
| **5** | *"What is the story behind the café and how did it start?"* | `menu_drinks.txt` (0.1902) / `story.txt` (0.1946) | 0.1902 | **Highly Relevant:** Top-2 results contain the founder story (Mira Halloran) and the grandmother's kettle from Cork. |

---

### 3. Core Retrieval Concepts & Semantic vs. Keyword Search
- **Embedding Model:** Maps discrete words into continuous semantic vector spaces where conceptually related statements are positioned close to one another regardless of exact vocabulary.
- **Vector Database (ChromaDB):** An optimized engine for storing high-dimensional vectors and executing Approximate Nearest Neighbor (ANN) index searches (using algorithms like HNSW).
- **Similarity Search & Top-K Retrieval:** Ranks documents according to vector distance (e.g., Cosine Distance = $1 - \text{cosine similarity}$). Top-$K$ selects the $K$ smallest-distance items to construct the prompt context.
- **Semantic Search vs. Exact Keyword Matching:**  
  *Example:* The query *"I want something warm to drink because it is freezing outside"*.
  - **Exact Keyword Matching:** Returns 0 hits because the words *"warm"*, *"freezing"*, and *"outside"* do not appear in `menu_drinks.txt`.
  - **Semantic Vector Search:** Successfully retrieves `menu_drinks.txt` (distance 0.165) because the multilingual E5 model understands the conceptual similarity between cold weather/warm drinks and coffee, tea, and drinking chocolate.

---

### 4. Retrieval Improvement: Semantic Section Chunking with Metadata
- **Proposed Improvement:** The baseline system indexes each entire `.txt` file as one single document (1,200–1,700 characters). This dilutes specific facts across irrelevant sections. The improved implementation (`steps/01_retrieval_improvement.py`) parses documents into semantic section chunks and attaches rich metadata (`source`, `section`, `char_length`).
- **Empirical Comparison:**

| Query | Baseline Document | Baseline Dist | Baseline Payload | Improved Chunk ID | Improved Dist | Improved Payload | Compression & Improvement |
|---|---|---|---|---|---|---|---|
| *"Can I bring pets or animals?"* | `faq.txt` | 0.1860 | 1,651 chars | `faq.txt#chunk_6` (ANIMALS) | **0.1620** | **196 chars** | **88.1% context reduction**, distance improved by 0.024 |
| *"Are there power outlets for laptops?"* | `faq.txt` | 0.2276 | 1,651 chars | `faq.txt#chunk_3` (POWER OUTLETS) | **0.1799** | **198 chars** | **88.0% context reduction**, distance improved by 0.048 |
| *"Do you have peanut allergies or use peanuts?"* | `allergens.txt` | 0.1948 | 1,627 chars | `allergens.txt#chunk_4` (NUTS) | **0.1470** | **265 chars** | **83.7% context reduction**, distance improved by 0.048 |
| *"What time do you close on Sunday?"* | `hours_access.txt` | 0.1973 | 1,286 chars | `hours_access.txt#chunk_1` (OPEN HOURS) | **0.1338** | **441 chars** | **65.7% context reduction**, distance improved by 0.064 |

**Conclusion:** Semantic chunking drastically reduces noise, cuts context window consumption by up to 88%, lowers vector distances, and isolates the exact paragraphs needed for reasoning.

---

## Part B: Expose Retrieval Through MCP (Model Context Protocol)

### 1. Protocol Architecture & The Initialization Exchange
The Model Context Protocol standardizes how host applications connect LLMs to data and tools over standard communication channels (such as JSON-RPC 2.0 over `stdio`).
- **Standard In/Out Transport:** The client spawns the server as a subprocess (`uv run python -m server.main`), sending newline-delimited JSON-RPC requests via `stdin` and reading responses from `stdout`. Diagnostic logs are kept strictly on `stderr`.
- **Initialization Handshake:**
  1. Client sends `initialize` request specifying protocol version, capabilities, and client metadata.
  2. Server responds with its protocol version, capabilities (e.g. tools, resources, prompts), and server info.
  3. Client acknowledges with a `notifications/initialized` notification (no `id`, no response expected).

```json
// [1] Client -> Server: initialize
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "raw-client", "version": "0.1.0"}}}

// [2] Server -> Client: initialize response
{"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {"tools": {"listChanged": false}}, "protocolVersion": "2025-06-18", "serverInfo": {"name": "rag-search-server", "version": ""}}}

// [3] Client -> Server: notifications/initialized
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

---

### 2. Exposed Tools & Purpose of `tools/list` and `tools/call`
- `tools/list`: Allows the client to dynamically discover what functions the server provides along with their JSON Schema parameters (`inputSchema`).
- `tools/call`: Executes a specific tool with parameter arguments validated against its schema.

The server (`src/server/main.py`) exposes three distinct tools:
1. `search(query: str, top_k: int = 5)`: Vector search over the café documents.
2. `list_documents()`: Returns the list of document filenames.
3. `get_document(name: str)`: Fetches an entire document by exact filename.

---

### 3. Sanitized Request / Response Tool Trace
**Request from Client to Server:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "can I bring a dog",
      "top_k": 1
    }
  }
}
```

**Response from Server to Client:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[1] source: faq.txt (distance: 0.21)\npassage:\nANIMALS\nDogs are welcome on the small terrace at the top of the stairs, but not inside the room itself. There is a water bowl by the terrace door. Assistance animals are welcome anywhere."
      }
    ]
  }
}
```

---

### 4. Distinguishing MCP from the Tool's Underlying Functionality
**MCP does not perform the search.** MCP is strictly a **communication and interoperability protocol**. It specifies message structures, capabilities, and schema negotiation so any LLM client (Claude, Ollama, custom agents) can discover and invoke functions uniformly. The underlying functionality (text chunking, sentence embeddings, ChromaDB vector indexing) is decoupled and executed independently behind the MCP tool interface.

---

## Part C: Connect the LLM and RAG

### 1. End-to-End Sequence
```
[User Question]
       │
       ▼
[LLM Tool Decision] ──► Inspects question + MCP tool definitions -> outputs tool call: search(query="...")
       │
       ▼
[MCP Tool Invocation] ──► Client sends JSON-RPC tools/call -> MCP Server executes search in ChromaDB
       │
       ▼
[Retrieved Context] ──► Relevant passages returned over stdio to Client
       │
       ▼
[Final LLM Response] ──► Context + Question passed to LLM -> LLM generates grounded answer
```

---

### 2. Evaluation Across Five Café Topics
1. **Opening Hours:** *"What time does the café open and close on Saturday and Sunday?"*  
   - *Retrieved:* `hours_access.txt` (Saturday: 9:00 to 22:00, Sunday: 9:00 to 17:00).  
   - *Grounded Answer:* The café opens at 9:00 on Saturday and closes at 22:00; on Sunday it opens at 9:00 and closes at 17:00.
2. **Food & Dietary:** *"What options do you have for someone who cannot eat wheat or gluten?"*  
   - *Retrieved:* `allergens.txt`.  
   - *Grounded Answer:* The café explicitly notes it has **no gluten-free bread** and all pastries contain wheat. Rice porridge and daily soup are usually safe (check daily for barley stock).
3. **Working Environment:** *"Can I work on my laptop there, and are there power outlets available?"*  
   - *Retrieved:* `faq.txt`.  
   - *Grounded Answer:* Yes, laptops are welcome on most days with one order per 90 minutes. Laptops are put away Friday and Saturday after 6:00 PM. Six sockets are located along the window bench and two near the books.
4. **Directions & Arrival:** *"How do I reach the café from Shibuya and where can I park a bicycle?"*  
   - *Retrieved:* `hours_access.txt`.  
   - *Grounded Answer:* 15-minute train ride from Shibuya to Kitasawa station on the Inokashira line, followed by a 6-minute walk. Bicycle racks for ~8 bikes are located beside the green door.
5. **Origin Story:** *"Who founded The Wandering Kettle and what is the meaning behind its name?"*  
   - *Retrieved:* `story.txt`.  
   - *Grounded Answer:* Founded by Mira Halloran. Named after an ornamental, non-functional kettle inherited from her grandmother's house in Cork.

---

### 3. Unanswerable Question Test & Safeguards
- **Test Question:** *"Do you sell branded ceramic mugs, t-shirts, or tote bags as café merchandise?"*
- **Outcome with System Safeguard Prompt:**  
  *Retrieved Context:* `data/documents/` does not contain merchandise records.  
  *LLM Output:* **"The documents on file do not contain information about what merchandise the café sells."**
- **Vulnerability without Safeguard:** When unconstrained, small LLMs tend to fabricate details (e.g. *"Yes, mugs are available at the counter for $12"*).
- **Recommended Safeguards:**
  1. **Prompt Guard:** Explicit negative constraint: *"Answer strictly using the provided context. If the answer is absent from the text, state clearly that the information is unavailable. Do not assume or extrapolate."*
  2. **Application-level Distance Threshold:** If the closest retrieved document has a vector distance exceeding a predefined threshold (e.g. $\text{distance} > 0.35$), the application rejects the query before calling the LLM: *"No relevant café records found."*

---

## Part D: Extension: Develop an Iterative ReAct Agent

### 1. ReAct Loop Implementation Architecture
Implemented in Python in `steps/05_react_agent.py` without heavy third-party agent frameworks:
```python
while step < MAX_STEPS:
    decision = ask_llm(messages, available_tools)
    if decision is final_answer:
        return decision
    result = execute_selected_mcp_tool(decision)
    messages += [tool_call, tool_result]
return safe_stop_message
```

---

### 2. Challenge Request Execution Trace
**User Request:**  
> *"I'm planning to visit the café with my cat. Can I sit outside, what food options are available, and are there any allergy considerations I should know about?"*

```
================================================================================
STARTING ITERATIVE ReAct AGENT
================================================================================

--- ITERATION 1 ---
[Thought / Reasoning]:
  The user is asking three separate things: pet policy for sitting outside, food options,
  and allergy considerations. I will first check the pet and outdoor seating policy.
[Action]: Call Tool 'search'
  Arguments: {"query": "can I bring a cat outside pet policy", "top_k": 1}
[Observation] (1,689 characters received from faq.txt):
  ANIMALS: Dogs are welcome on the small terrace at the top of the stairs, but not
  inside the room itself. There is a water bowl by the terrace door. Assistance
  animals are welcome anywhere.
[Next Decision]: Policy found for dogs and terrace, but food options and allergies are missing.

--- ITERATION 2 ---
[Thought / Reasoning]:
  Now I need to find the available food menu options to answer the second part.
[Action]: Call Tool 'search'
  Arguments: {"query": "food menu options all day plates sandwiches", "top_k": 1}
[Observation] (1,252 characters received from menu_food.txt):
  ALL DAY PLATES: Scrambled eggs on sourdough ($6.20), Mushroom toast ($6.60),
  Soup of the day ($5.40), Rice porridge ($5.00). SANDWICHES: Ham and comte ($5.80),
  Roasted vegetable hummus ($5.60), Egg salad ($4.90), Tuna cucumber ($5.40).
  PASTRY CASE: Croissant, Almond croissant, Cinnamon roll, Lemon poppy seed cake, Banana bread.
[Next Decision]: Food menu identified. I still need allergy and dietary considerations.

--- ITERATION 3 ---
[Thought / Reasoning]:
  I must check the allergy policies and dietary restrictions to answer the third part.
[Action]: Call Tool 'search'
  Arguments: {"query": "allergy considerations dietary ingredients cross contamination", "top_k": 1}
[Observation] (1,671 characters received from allergens.txt):
  WITHOUT MILK: Oat and soy milk available at no extra charge. Drinking chocolate contains dairy.
  WITHOUT WHEAT: We have no gluten free bread. Pastries contain wheat.
  NUTS: Almond croissants handled on same surface; no peanuts used in kitchen.
  SHARED KITCHEN: One small surface, cannot guarantee any item free of traces.
[Next Decision]: Sufficient evidence collected for all three parts. Synthesizing final answer.

--- ITERATION 4 (FINAL ANSWER) ---
[Thought / Reasoning]:
  I have verified all three parts from official documents. I will distinguish confirmed
  facts (dogs on terrace, food items, cross-contamination) from unconfirmed rules (cats).
[Decision]: Final Answer reached.
```

**Final Synthesized Answer:**
> **1. Outdoor Seating & Cat Policy:** You can sit outside on the small terrace at the top of the stairs. However, the café's official policy explicitly welcomes **dogs** on the terrace (with a water bowl provided) and assistance animals anywhere. The documents do **not** explicitly state whether cats are permitted.  
> **2. Food Options:** The café offers All-Day Plates (scrambled eggs on sourdough \$6.20, mushroom toast \$6.60, soup of the day \$5.40, rice porridge \$5.00), made-to-order Sandwiches (ham & comte, roasted vegetable & hummus, egg salad, tuna cucumber), and fresh pastries (croissants, almond croissants, cinnamon rolls, lemon poppy seed cake, and house-made banana bread).  
> **3. Allergy Considerations:** The kitchen operates with a single preparation surface and oven, so **no item can be guaranteed free of trace allergens**. The café has **no gluten-free bread**, and all pastries contain wheat. Nut allergies must be cautious as almond pastries share preparation surfaces (though peanuts are never used). Oat and soy milk are offered at no extra charge.

---

### 3. One-Shot Tool Call vs. Iterative ReAct Loop
- **One-Shot Tool Call:** The LLM issues a single query (e.g. `search("cat outside food allergies")`), which returns either one muddled document or hits context limits. The model cannot inspect results, follow up on missing pieces, or gather multi-domain evidence.
- **Iterative ReAct Loop:**
  - **Reasoning (Thought):** Evaluates what sub-goals have been answered and identifies what facts remain unknown.
  - **Action (Decision):** Chooses the optimal tool and specific arguments targeting only the missing evidence.
  - **Observation:** Ingests the tool output, updates conversational history, and loops back to Reasoning.

---

### 4. Agent Limitations & Safeguard Engineering
- **Observed Limitation 1: Guessing Non-Existent Filenames.**  
  *Observation:* When given `get_document`, the model initially guessed non-existent filenames like `allergies.txt` or `menu.txt`.  
  *Safeguard:* Strict prompt instruction to use `search` or call `list_documents` first, combined with application-level error catching returning informative error messages back into the observation history.
- **Observed Limitation 2: Premature Termination.**  
  *Observation:* Under lightweight models, the LLM may stop after the first search and fabricate the remaining questions.  
  *Safeguard:* An application-level multi-step coordinator that inspects whether composite sub-questions have been queried before accepting termination.

---

## Reproducibility & Setup Instructions

### Option 1: Running in Docker (Recommended)
```bash
# Navigate to project directory
cd mcp-rag-from-scratch

# Setup environment variables
cp .env.example .env

# Build and start services in detached mode
docker compose up --build -d

# Run all unit and integration tests
docker compose exec app uv run pytest
```

### Option 2: Running Directly on Host with `uv`
```bash
# Ensure local Ollama is active
ollama run qwen2.5:1.5b

# Run full test suite (17 passed)
uv run pytest

# Run Step 1 (Vector DB)
uv run python steps/01_chromadb.py

# Run Retrieval Improvement Comparison
uv run python steps/01_retrieval_improvement.py

# Run Step 2 (Raw JSON-RPC MCP)
uv run python steps/02_mcp_raw.py

# Run Step 3 (MCP Official SDK)
uv run python steps/03_mcp_sdk.py

# Run Step 4 (LLM RAG)
uv run python steps/04_rag.py

# Run Part D Iterative ReAct Agent
uv run python steps/05_react_agent.py

# Run Automated Evaluation Suite for Parts A & C
uv run python steps/demo_runner.py
```
