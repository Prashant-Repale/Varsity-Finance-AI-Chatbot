<div align="center">

# 📈 Varsity Finance AI

### An Agentic RAG Financial Chatbot — Markets, News & Investing Education in One Chat

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/Groq-LLM_Inference-F55036?style=for-the-badge&logo=groq&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-Memory-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">
</p>

</div>

<br>

<div align="center">
  <img src="C:\Users\prash\OneDrive\Pictures\Screenshots\Screenshot (13).png" alt="Varsity Finance AI demo" width="850">
  <br>
  <sub><i>👆 Replace this with a screen recording / screenshot of your chat UI</i></sub>
</div>

<br>

## 🌟 Overview

**Varsity Finance AI** is a production-style **agentic RAG chatbot** for stock markets and investing education. It blends a curated knowledge base (all **17 Zerodha Varsity modules**) with **live market data** and **real-time web search**, wrapped in a sleek ChatGPT-style interface — and it *remembers* who you are across sessions.

|  |  |
|---|---|
| 🧠 **Agentic core** | LangGraph `StateGraph` decides *when* and *which* tool to call |
| 📚 **Grounded answers** | RAG over Zerodha Varsity using ChromaDB + HuggingFace embeddings |
| 📈 **Live markets** | Real-time prices, P/E, market cap & 52-week range via `yfinance` |
| 🌐 **Fresh news** | Real-time web search via Tavily for breaking events |
| 🗄️ **Dual memory** | PostgreSQL-backed short-term (per-chat) + long-term (per-user) memory |
| 🎨 **Polished UI** | Dark, ChatGPT/Codex-style Streamlit interface with chat history sidebar |

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
```

<br>

## 🔧 Tools at a Glance

| Tool | Powered By | Used For |
|---|---|---|
| `rag_tool` | ChromaDB + HuggingFace embeddings | Finance concepts & education (Zerodha Varsity — 17 modules) |
| `get_stock_info` | `yfinance` | Live price, P/E, market cap, 52-week high/low, volume |
| `web_search` | Tavily | Latest news, events & market updates |

## 🧠 Memory System

| Type | Storage | Purpose |
|---|---|---|
| **Short-term** | `PostgresSaver` (checkpoints) | Per-thread conversation state; auto-summarized once a chat grows past 6 messages |
| **Long-term** | `PostgresStore` | Persistent user profile (name, goals, risk profile, interests) — extracted automatically and reused silently |

<br>

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|:---:|:---:|
| **Frontend** | Streamlit |
| **Agent Orchestration** | LangGraph (`StateGraph`) |
| **LLM Inference** | Groq — `gpt-oss-120b` |
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

<br>

## 🗺️ Roadmap

- [x] Agentic tool routing (RAG · live stock data · web search)
- [x] Short-term + long-term memory via PostgreSQL
- [x] ChatGPT-style UI with multi-thread chat history
- [ ] Portfolio tracking dashboard
- [ ] Voice input / output
- [ ] One-click deploy (Docker Compose / Streamlit Cloud)

<br>

## 🙏 Acknowledgements

- **[Zerodha Varsity](https://zerodha.com/varsity/)** — the educational backbone of the knowledge base
- **[LangChain / LangGraph](https://www.langchain.com/)** — agent orchestration framework
- **[Groq](https://groq.com/)** — blazing-fast LLM inference

<br>

## 📄 License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ as a final-year engineering project</sub>
</div>