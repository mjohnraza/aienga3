# Video Demonstration Script & Walkthrough Guide
## Practical Assignment 3: Knowledge Assistant Bot (The Wandering Kettle)
**Duration:** ~5–8 minutes  
**Format:** Screen Recording with Voiceover (showing Terminal + Visual Diagram / Code)

---

### Introduction (0:00 - 0:45)
- **Visual:** Open VS Code / Terminal with the project repository `mcp-rag-from-scratch`.
- **Spoken:**
  > *"Hello! In this demonstration, I present my implementation for Practical Assignment 3: Building a Knowledge Assistant Bot for the fictional café 'The Wandering Kettle'. We cover the progression from vector retrieval with ChromaDB, through the Model Context Protocol (MCP) using both raw JSON-RPC and the Python SDK, to LLM-enabled RAG, and ultimately an iterative ReAct agent capable of multi-step reasoning."*

---

### Part A: RAG Pipeline & ChromaDB (0:45 - 2:00)
- **Visual:** Run in terminal:
  ```bash
  uv run python steps/01_chromadb.py
  uv run python steps/01_retrieval_improvement.py
  ```
- **Spoken Points to Address:**
  1. **Ingestion & Vectors:** Show that 7 source text files are loaded, embedded via `intfloat/multilingual-e5-small` into 384-dimensional dense vectors, and stored in ChromaDB.
  2. **Similarity Search:** Explain how query vectors are matched using cosine distance. Point out the test where the query *"I want something warm to drink because it is freezing outside"* successfully retrieves `menu_drinks.txt` even though the words "warm" and "freezing" never appear in the text—proving semantic understanding over keyword matching.
  3. **Retrieval Improvement:** Show the before-and-after comparison in `steps/01_retrieval_improvement.py`. Explain how chunking documents by section headers combined with metadata cuts payload sizes by over 80% while significantly improving vector distance precision.

---

### Part B: Exposing Retrieval Through MCP (2:00 - 3:30)
- **Visual:** Run in terminal:
  ```bash
  uv run python steps/02_mcp_raw.py
  uv run python steps/03_mcp_sdk.py
  ```
- **Spoken Points to Address:**
  1. **Raw JSON-RPC over stdio:** Explain the three-phase initialization handshake (`initialize` -> response -> `notifications/initialized`).
  2. **Tool Discovery (`tools/list`):** Show the server returning its three exposed tools (`search`, `list_documents`, `get_document`) and their input schemas.
  3. **Tool Invocation (`tools/call`):** Trace the exact JSON payload sent to invoke `search` and the structured JSON-RPC result containing the text block returned.
  4. **Key Distinction:** Emphasize that *MCP does not perform the search itself*; it standardizes how clients discover and call tools across process boundaries.

---

### Part C: Connect the LLM and RAG (3:30 - 4:45)
- **Visual:** Run in terminal:
  ```bash
  uv run python steps/04_rag.py
  uv run python steps/demo_runner.py
  ```
- **Spoken Points to Address:**
  1. **End-to-End Flow:** Explain the flow: User question -> LLM decides to call `search` -> MCP server executes retrieval in ChromaDB -> Retrieved context injected into prompt -> LLM produces grounded answer.
  2. **Demonstrate Cafe Topics:** Show answers to opening hours, vegan/gluten-free options, working policies, and the founder's story.
  3. **Unanswerable Question Safeguard:** Show the test question *"Do you sell branded ceramic mugs or merchandise?"*. Point out that with the system safeguard, the assistant accurately states that the café documents do not contain merchandise information, rather than hallucinating an answer.

---

### Part D: Iterative ReAct Agent (4:45 - 6:30)
- **Visual:** Run in terminal:
  ```bash
  uv run python steps/05_react_agent.py
  ```
- **Spoken Points to Address:**
  1. **Challenge Request:**
     > *"I'm planning to visit the café with my cat. Can I sit outside, what food options are available, and are there any allergy considerations I should know about?"*
  2. **Multi-Iteration Trace Walkthrough:**
     - **Iteration 1:** Agent reasons that pet/terrace policy is needed, calls `search(query="pet policy animal outdoor terrace")`, and retrieves `faq.txt`.
     - **Iteration 2:** Observes dog terrace policy, reasons food menu is missing, calls `search(query="food menu all day plates")`, and retrieves `menu_food.txt`.
     - **Iteration 3:** Observes food options, reasons allergy notes are missing, calls `search(query="allergy considerations dietary")`, and retrieves `allergens.txt`.
     - **Iteration 4 (Final Synthesis):** Agent evaluates that all three components have verified evidence and outputs the comprehensive final response.
  3. **Distinguishing Confirmed vs. Missing Policy:** Show that the agent confirms terrace seating for dogs and assistance animals, but explicitly flags that the café policy makes no mention of cats!
  4. **Agent Limitation & Safeguard:** Discuss agent limitations (premature termination or guessing non-existent file names) and show how the application-level loop and schema validation safeguard the execution.

---

### Conclusion & Tests (6:30 - 7:00)
- **Visual:** Run in terminal:
  ```bash
  uv run pytest
  ```
- **Spoken:**
  > *"All 17 tests pass successfully across ChromaDB, the indexer, the MCP server, and client integrations. Everything runs 100% locally with Docker or native uv. Thank you!"*
