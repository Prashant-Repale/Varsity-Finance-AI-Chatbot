<div align="center">

# 📈 Varsity Finance AI

### An Agentic RAG Financial Chatbot — Markets, News & Investing Education in One Chat

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
  <img alt="LangSmith" src="https://img.shields.io/badge/LangSmith-Tracing-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/Groq-LLM_Inference-F55036?style=for-the-badge&logo=groq&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-Memory-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">
</p>

</div>

<br>

<div align="center">
  <img src="docs/demo.gif" alt="Varsity Finance AI demo" width="850">
  <br>
  <sub><i>👆 Replace this with a screen recording / screenshot of your chat UI</i></sub>
</div>

<br>

## 🌟 Overview

> Most finance chatbots either answer from frozen training data — and get prices wrong — or bolt one API onto a chat loop with no fallback logic. **Varsity Finance AI treats every query as a routing decision**: tool use is enforced (not suggested) by the system prompt, retrieval quality is validated before answering, and the agent manages its own context budget so long conversations don't blow past API rate limits.

Built on a LangGraph `StateGraph`, grounded in the **17 Zerodha Varsity modules**, and traced end-to-end with **LangSmith**.

|  |  |
|---|---|
| 🧠 **Agentic core** | LangGraph `StateGraph` decides *when* and *which* tool to call — never both at once |
| 📚 **Grounded answers** | RAG over Zerodha Varsity (ChromaDB + HuggingFace embeddings), gated by a relevance check |
| 📈 **Live markets** | Real-time prices, P/E, market cap & 52-week range via `yfinance` |
| 🌐 **Fresh news** | Real-time web search via Tavily, scoped to finance-domain queries only |
| 🗄️ **Dual memory** | PostgreSQL-backed short-term (per-chat) + long-term (per-user) memory |
| 🔍 **Full observability** | Every LLM call, tool call & token spend traced in **LangSmith** |
| 🎨 **Polished UI** | Dark, ChatGPT/Codex-style Streamlit interface with multi-thread history |

<br>

## ⚖️ Why Finance Is a Hard Domain for LLMs

| Challenge | Risk if Ignored | How This System Handles It |
|---|---|---|
| **Correctness** | A hallucinated number reads identically to a real one — but drives real decisions | Finance concepts are answered *only* from retrieved Varsity context; the model is explicitly forbidden from using prior knowledge |
| **Real-time data** | Prices, indices & news shift by the second; model training data is always stale | Live tools (`yfinance`, Tavily) are *mandatory* for time-sensitive queries — never answered from memory |
| **Reliability under constraints** | Free-tier rate limits & flaky tool calls can silently degrade answer quality | Token-budgeted prompts, explicit tool-failure messaging, and graceful fallback instead of silent guessing |

<br>

## 🧭 How It Works

```mermaid
flowchart LR
    Start(["👤 User Message"]) --> Chat["🧠 chat_node\nLLM + System Prompt"]
    Chat -- "tool needed" --> Tools["🔧 tools"]
    Tools --> Chat
    Chat -- "answer ready" --> Memory["🗂️ memory_node\nSummarize + Save Facts"]
    Memory --> End(["💬 Response"])

    Tools -.-> RAG[("📚 rag_tool\nChromaDB · Varsity KB")]
    Tools -.-> Stock[("📈 get_stock_info\nyfinance")]
    Tools -.-> Web[("🌐 web_search\nTavily")]
```

```mermaid
flowchart TB
    User["🧑‍💻 User"] <--> UI["🖥️ Streamlit UI\napp.py"]
    UI <--> Agent["🤖 LangGraph Agent\nbackend.py"]
    Agent <--> LLM["⚡ Groq LLM\ngpt-oss-120b"]
    Agent <--> Vector[("🧩 ChromaDB\nZerodha Varsity KB")]
    Agent <--> PG[("🗄️ PostgreSQL\nCheckpointer + Store")]
    Agent <--> Ext["🌐 External APIs\nyfinance · Tavily"]
    Agent -.-> LS[("🔍 LangSmith\nTracing & Evals")]
```

<sub>Every node and tool call in this graph is traced via LangSmith — useful for debugging tool-routing decisions and tracking token spend per turn.</sub>

<br>

## 🛠️ System Design Highlights

### 1. Tool Enforcement — Routing Is a Rule, Not a Suggestion

| Query Type | Required Tool | Hard Rule |
|---|---|---|
| Finance concept / education | `rag_tool` | Must answer *only* from retrieved context — refuses if context is insufficient |
| Live price, index, P/E, market cap | `get_stock_info` | `web_search` is explicitly disallowed for these — prevents stale or contradictory numbers |
| Recent news / events | `web_search` | Triggered only for finance-domain queries |
| Non-finance | *none* | Declines politely — keeps scope (and token spend) tight |

The system prompt encodes what the model **must not** do as explicitly as what it must — the primary anti-hallucination mechanism, not a post-hoc filter.

### 2. Token Optimization — Designed Around LLM Rate Limits

- Groq's rate limits cap throughput per minute, shared across the system prompt, tool outputs, and conversation history.
- The system prompt was rewritten from a verbose, rule-by-rule draft into a dense, enforceable ruleset — cut significantly with zero loss in routing logic.
- LangSmith traces surfaced that raw tool outputs (yfinance JSON, search snippets) were the largest context consumers heading into summarization.
- `filter_clean_messages()` strips tool-call and tool-result messages **before** the conversation is summarized — only user/assistant turns count toward the rolling summary.

### 3. Failure Handling — Insufficient RAG Context

`rag_tool` doesn't just return "no results." Below a similarity threshold, it returns a structured signal that tells the agent:

1. Check whether the question is even in the finance domain.
2. If yes → escalate to `web_search`.
3. If no → decline politely, with **no further tool calls**.

This avoids the two failure modes that plague RAG bots — answering anyway (hallucination) or dead-ending with "I don't know."

### 4. Memory Architecture — Two Tiers, One Store

| Type | Storage | Behavior |
|---|---|---|
| **Short-term** | `PostgresSaver` (checkpoints) | Full state per thread. Past 6 messages, older turns are summarized into a 3–4 sentence rolling summary and removed via `RemoveMessage`, keeping only the last 2 exchanges + summary in context |
| **Long-term** | `PostgresStore` | A lightweight, Pydantic-parsed extraction call pulls durable facts (name, goals, risk profile, interests) per `user_id` and silently injects them into future prompts |

<br>

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|:---:|:---:|
| **Frontend** | Streamlit |
| **Agent Orchestration** | LangGraph (`StateGraph`) |
| **LLM Inference** | Groq — `gpt-oss-120b` |
| **Observability** | LangSmith |
| **Vector Database** | ChromaDB |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Persistence** | PostgreSQL |
| **Market Data** | yfinance |
| **Web Search** | Tavily |

</div>

<br>

## 📂 Project Structure

```
.
├── app.py              # Streamlit chat UI — sidebar, threads, streaming
├── backend.py          # LangGraph agent — tools, memory, system prompt
├── chroma_db/          # Persisted vector store (Zerodha Varsity KB)
├── requirements.txt    # Python dependencies
└── .env                # API keys & DB connection string (not committed)
```

<br>

## ⚙️ Quick Start

**1. Clone & install**

```bash
git clone https://github.com/<your-username>/varsity-finance-ai.git
cd varsity-finance-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Spin up PostgreSQL** (used for chat memory + threads)

```bash
docker run --name varsity-pg -e POSTGRES_PASSWORD=postgres -p 5442:5432 -d postgres:16
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5442/postgres
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
PYTHONIOENCODING=utf-8

# Optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=rag_finance_project
```

**4. Run the app**

```bash
streamlit run app.py
```

Then open **http://localhost:8501** 🎉

<br>

## 🔑 Environment Variables

| Variable | Required | Description |
|---|:---:|---|
| `DATABASE_URL` | ✅ | PostgreSQL connection string for checkpointer & long-term store |
| `GROQ_API_KEY` | ✅ | API key for Groq-hosted LLM |
| `TAVILY_API_KEY` | ✅ | API key for real-time web search |
| `PYTHONIOENCODING` | ✅ | Ensures consistent UTF-8 I/O across platforms |
| `LANGCHAIN_TRACING_V2` | ⚪ | Enables LangSmith tracing (`true` / `false`) |
| `LANGCHAIN_ENDPOINT` | ⚪ | LangSmith API endpoint (region-specific) |
| `LANGCHAIN_API_KEY` | ⚪ | API key for LangSmith |
| `LANGCHAIN_PROJECT` | ⚪ | LangSmith project name for trace grouping |

<br>

## 🙏 Acknowledgements

- **[Zerodha Varsity](https://zerodha.com/varsity/)** — the educational backbone of the knowledge base
- **[LangChain / LangGraph](https://www.langchain.com/)** — agent orchestration framework
- **[Groq](https://groq.com/)** — high-throughput LLM inference

<br>

## 📄 License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ as a final-year engineering project</sub>
</div>