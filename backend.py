from typing import Annotated, TypedDict
import os  
import yfinance as yf
from dotenv import load_dotenv
# from duckduckgo_search import DDGS
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage ,AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field
import streamlit as st
from datetime import datetime
import psycopg
import sys
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
load_dotenv()
os.environ["PYTHONIOENCODING"] = "utf-8"

DATABASE_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
groq_key     = os.getenv("GROQ_API_KEY")  or st.secrets.get("GROQ_API_KEY")
tavily_key   = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")


MODEL_NAME = "openai/gpt-oss-120b"
DB_URI = DATABASE_URL
CHROMA_PATH = "chroma_db"

# model = ChatGroq(
#     model=MODEL_NAME,
#     temperature=0,
#     max_tokens=1024,
# )
# 🔹 CACHE LLM (optional but good)
@st.cache_resource
def load_llm():
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=1000,
    )

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
# )
# 🔹 CACHE EMBEDDINGS (IMPORTANT)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# db = Chroma(
#     persist_directory=CHROMA_PATH,
#     embedding_function=embeddings,
# )

# 🔹 CACHE VECTOR DB (IMPORTANT)
@st.cache_resource
def load_vector_db(_embeddings):
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=_embeddings,
    )

# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 5,
#         "fetch_k": 30,
#         "lambda_mult": 0.7,
#     },
# )

# 🔹 CACHE RETRIEVER (IMPORTANT)
@st.cache_resource
def load_retriever(_db):
    return _db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },
    )

model = load_llm()
embeddings = load_embeddings()
db = load_vector_db(embeddings)
retriever = load_retriever(db)


@tool
def rag_tool(query: str) -> dict:
    """
    Use this tool ONLY for conceptual or educational finance questions.
    Do NOT use this for live stock prices, recent news, or math calculations.
    """
    
    results = retriever.invoke(query)
    relevance_check = db.similarity_search_with_relevance_scores(query, k=1)

    if not relevance_check or relevance_check[0][1] < 0.5:
        return {
            "status": "find out the domain of the question is it finance or not finance ",
            "instruction": "No docs found in knowledge base.if the question is of finance domain and you dont have relavant information from this tool then send it to web search tool only if it is finance doman , if question is out of the finance doman Use NO tools say polietly Call web_search instead.",
            "context": [],
        }
    context = [doc.page_content for doc in results]
    metadata = [doc.metadata for doc in results]
    return {"query": query, "context": "\n\n".join(context), "metadata": metadata}


@tool
def get_stock_info(ticker: str) -> dict:
    """
    Use this tool for live/recent stock data: price, P/E ratio, market cap,
    52-week high/low, day high/low, dividends.
    Good examples:
      - "What is TCS stock price today?"
      - "Reliance Industries market cap"
      - "Apple P/E ratio"
      - "Infosys 52-week high"
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "company_name": info.get("longName"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


from tavily import TavilyClient
TODAY = datetime.now().strftime("%B %d, %Y")
@tool
def web_search(query: str) -> str:
    """
    Use this tool for recent news, latest events, current market updates,
    recent company announcements, or any information that needs the internet, todays date is { TODAY}.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True
        )
        formatted = []
        if response.get("answer"):
            formatted.append(f"Quick Summary: {response['answer']}")
        for r in response.get("results", []):
            content_preview = r.get('content', '')[:400]
            formatted.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Summary: {content_preview}\n"
                f"Source: {r.get('url', 'N/A')}"
            )
        return "\n\n---\n\n".join(formatted) if formatted else "No search results found for this query."
    except Exception as e:
        return f"Search failed: {str(e)}"


tools = [rag_tool, get_stock_info, web_search]
llm_with_tools = model.bind_tools(tools)



# SYSTEM_PROMPT = """You are a helpful financial assistant for Indian and global markets.
# You have access to 3 tools. Your job is to answer questions accurately using these tools when appropriate.
# Today's date is {TODAY}. Always use this date when searching for current news.
# When using web_search tool, always include the current year (2026) in your search query.
# ━━━━━━━━━━━
# ROUTING RULES:
# ━━━━━━━━━━━
# • rag_tool
#   → Use for finance concepts, definitions, and educational explanations

# • get_stock_info
#   → Use for live stock data: price, P/E, market cap, 52-week high/low

# • web_search
#   → Use for recent news, latest events, or current updates
# ━━━━━━━━━━━
# CRITICAL RULES:
# ━━━━━━━━━━━
# 1. For finance-related questions:
#    → You MUST use the appropriate tool before answering.
# 2. When using rag_tool:
#    → Base your answer ONLY on the retrieved context
#    → Do NOT add external knowledge ,If context is insufficient, say: "I don’t have enough information in the knowledge base to answer this fully."
# 3. For non-finance questions:
#    → Do NOT use any tool ,Politely say the question is outside your domain
# 4. After using any tool:
#    → Do NOT return raw tool output
#    → Interpret and explain clearly in simple language
# 5.For financial indicators like:
# - VIX ,indices (Nifty, Sensex) ,stock prices ,ALWAYS use get_stock_info tool
# → DO NOT use web_search
# 6.If a tool fails :
# - Do NOT generate a generic answer
# - Clearly mention the failure
# - Provide alternative guidance,Suggest how user can proceed, Keep answer helpful and actionable
# 7.If a question is vague or ambiguous:
# - Do NOT answer directly
# - Ask a short and natural clarification question
# - Do NOT mention tool failures or system limitations
# - Keep the tone conversational, not formal
# 8. FORMATTING RULES:
# - Do NOT use LaTeX notation like \\text{}, \\frac{}, \\sqrt{} etc.
# - Write formulas in plain text. Example:
#   PAT = PBT (Profit Before Tax) - Taxes
#   EPS = PAT / Total Outstanding Shares
# - Use simple markdown: **bold**, bullet points, and numbered lists only.
# 9. RESPONSE LENGTH CONTROL:
# - Keep the main answer concise (max ~600-700 tokens)
# - Always reserve space for follow-up questions and sources
# - If answer is long, summarize instead of expanding
# 10. ENDING SECTION (MANDATORY):
# - Always include follow-up questions and sources
# - If needed, shorten the explanation to ensure this section appears
# -At the end of your answer always , include:
# -.would you like to ask :
# -give 3 short, contextual, high-engagement follow-up questions before printing the sources
#   Source:
# - List the sources and which tool used from the retrieved data
# - Do NOT mention tool failures or internal system issues to the user
# - if u dont have any source then don't mention it, also word source  .

# DISCLAIMER:
# "Append this only when user is asking for risk related suggestion '⚠️ This is for educational purposes only and not financial advice .' only if the query involves specific stocks, investment decisions, or advice; otherwise omit it."
# """

# we should reduce the token of system prompt , due to context window limitation , for groq api  , only 6k tonkens per minute 
SYSTEM_PROMPT = """ **Role:** Financial assistant for Indian/global markets. Date: {TODAY}.  
For time-sensitive queries, include "2026" in `web_search` queries.

**STRICT KNOWLEDGE RULE:**  
- You are NOT allowed to answer finance concepts from internal knowledge.  
- You MUST use `rag_tool`   For finance concepts/explanations. Base answers ONLY on context; no external knowledge. If insufficient, output exactly: "I don't have enough information in the knowledge base to answer this fully."
- If no context is provided, respond EXACTLY:  
  "I don't have enough information in the knowledge base to answer this."  
- DO NOT use prior knowledge under any condition.  
  
**Execution Rules:**  
1. **Output:** Explain tool results simply; NEVER return raw tool output. Keep ≤200 words.  
2. **Failures:** If a tool fails, clearly state the failure and suggest actionable next steps (e.g., refine query, specify company).  
3. **Ambiguity:** Ask a short clarifying question in natural tone. Do NOT mention system limitations.  
4. **Formatting:** Use simple markdown (**bold**, bullets). Plain text formulas only (e.g., PAT = PBT - Taxes).  
5. **Length:** Aim for ≤200 words in the main body to leave space for the mandatory ending.  

**Tool Routing (Finance queries MUST use a tool first):**  
- `rag_tool`: For finance concepts/explanations. Base answers ONLY on retrieved context , if the question is of finance domain and you dont have relavant information from this tool then send it to web search tool only if it is finance doman , if question is out of the finance doman Use NO tools say polietly.  
- `get_stock_info`: For live prices, indices, P/E, market cap, 52‑wk high/low. NEVER use `web_search` for these.  
- `web_search`: For recent news/events.  
- *Non-finance queries:* Use NO tools; politely state it is outside your domain.  

**MANDATORY ENDING (STRICT FORMAT):**  
You MUST ALWAYS end your response EXACTLY in this format:

would you like to ask :
1.[Generate a relevant follow-up question based on the user's query]\n
2.[Generate another relevant follow-up question]\n
3.[Generate another relevant follow-up question]\n

Source : [List each useful source on a new line ]
**Disclaimer:** Append ONLY when the user asks for advice (e.g., "Should I invest in X?").  
For such queries, add:  
"⚠️ This is for educational purposes only and not financial advice."
"""
class FactExtraction(BaseModel):
    has_new_fact: bool = Field(description="Whether the message contains a stable user fact worth storing")
    fact: str | None = Field(default=None, description="The fact to store")
    category: str | None = Field(default=None, description="One of: name, preference, goal, background, risk_profile, location, interest")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


def _extract_json_object(text: str) -> str:
    """Return the first JSON object from model output that may include reasoning text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def chat_node(state: ChatState, config: RunnableConfig, store: BaseStore):
    # ALWAYS start fresh with your master system prompt (The Rulebook)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # =========================================================================
    # 1. READ LONG-TERM MEMORY (From PostgreSQL Store)
    # =========================================================================
    user_id = config["configurable"].get("user_id")
    if user_id:
        # Search the database for this specific user's saved facts
        namespace = (user_id, "personal_facts")
        # stored_memories = store.search(namespace)
        stored_memories = list(store.search(namespace))

        if stored_memories:
            # Format the database rows into clean bullet points
            facts_list = [f"• [{item.key}]: {item.value['fact']}" for item in stored_memories]
            compiled_facts = "\n".join(facts_list)

            # Inject it silently as a system instruction
            long_term_prompt = (
                "--- PERSISTENT USER PROFILE INFO ---\n"
                f"{compiled_facts}\n"
                "Use this profile to personalize your answers. Do not explicitly state that you are reading from a profile."
            )
            messages.append(SystemMessage(content=long_term_prompt))

    # =========================================================================
    # 2. READ SHORT-TERM MEMORY (From current session summary)
    # =========================================================================
    if state.get("summary"):
        summary_prompt = f"This is Context memory of previous conversations:\n{state['summary']}"
        messages.append(SystemMessage(content=summary_prompt))

    # =========================================================================
    # 3. BUILD AND INVOKE
    # =========================================================================
    # Extend with the active messages currently stored in the state
    
    messages.extend(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def filter_clean_messages(messages):
    clean = []
    for m in messages:
        if isinstance(m, HumanMessage):
            clean.append(m)
        elif isinstance(m, AIMessage) and not m.tool_calls:
            clean.append(m)
    return clean

def memory_summarize_delete(state: ChatState, config: RunnableConfig, store: BaseStore):
    msgs = state["messages"]
    if not msgs:
        return {}

    # 🔥 CLEAN MESSAGES
    clean_msgs = filter_clean_messages(msgs)

    # =========================================================================
    # STEP A:  LONG-TERM FACT EXTRACTION (Using LCEL Chain)
    # =========================================================================
    latest_user_message = ""
    for m in reversed(clean_msgs):
        if isinstance(m, HumanMessage):
            latest_user_message = m.content
            break

    if latest_user_message:
        # 2. Define the prompt template with input variables
        parser = PydanticOutputParser(pydantic_object=FactExtraction)

        prompt_template = PromptTemplate(
            template="""
        Extract one short, stable personal fact from "{user_message}" for long-term chatbot memory. Keep only durable info (name, background, interests, goals, investment preferences, risk tolerance, location) and ignore temporary requests. If found, return has_new_fact = true with a concise normalized fact and a category from [name, preference, goal, background, risk_profile, location, interest]; otherwise return has_new_fact = false.
        {format_instructions}
        """,
            input_variables=["user_message"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        try:
            # 4. Invoke the model, then parse the JSON object even if the model emits thinking text.
            extraction_response = (prompt_template | model).invoke({"user_message": latest_user_message})
            extraction_text = getattr(extraction_response, "content", str(extraction_response))
            fact_result = parser.parse(_extract_json_object(extraction_text))

            if fact_result.has_new_fact and fact_result.fact and fact_result.category:
                user_id = config["configurable"].get("user_id")
                if user_id:
                    store.put((user_id, "personal_facts"), fact_result.category, {"fact": fact_result.fact})
                    # print(f"📦 [LONG-TERM MEMORY] Saved to key '{fact_result.category}': {fact_result.fact}")

        except Exception as e:
            # General fallback to keep the chat running safely if formatting fails
            print(f"Long-term memory extraction skipped: {repr(e)}")
    if len(msgs) > 6:
        existing_summary = state.get("summary", "")

        if existing_summary:
            prompt = (
                f"Existing summary:\n{existing_summary}\n\n"
                "New conversation to merge in — be concise,over all summary should be  max 3-4 sentences: Preserve: user goals, preferences, unresolved tasks. Ignore: tool outputs, system messages")
        else:
            prompt = (
                "Summarize this conversation in max 3-4 sentences.Preserve: user goals, preferences, unresolved tasks. Ignore: tool outputs, system messages"
            )

        messages_for_summary = clean_msgs[:-4] + [HumanMessage(content=prompt)]
        response = model.invoke(messages_for_summary)
        new_summary = response.content

        # STRATEGY: Delete everything EXCEPT the last 4 messages (2 full chat turns)
        # This keeps the immediate context completely intact so the bot doesn't feel jumpy
        messages_to_delete = msgs[:-4]

        # Return BOTH updates simultaneously to the LangGraph state
        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
        }

    # If messages are less than 10, do absolutely nothing and let the graph finish
    return {}


tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("memory_node", memory_summarize_delete)

# 1. The entry point stays the same
graph.add_edge(START, "chat_node")
# 2. THE FIX: We map the condition.
# If tools -> go to tools. If END -> go to clean_up instead!
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", "__end__": "memory_node"})
# 3. Tools loop back to chat as normal
graph.add_edge("tools", "chat_node")
# 4. THE FINISH LINE: After cleanup finishes, we officially end the turn
graph.add_edge("memory_node", END)


def build_chatbot(checkpointer, store):
    """Compile and return the LangGraph chatbot."""
    return graph.compile(
        checkpointer=checkpointer,
        store=store,
    )


if __name__ == "__main__":
    welcome_message = """
        ======================================================================
        📈 Agentic RAG Financial Chatbot | Zerodha Varsity Knowledge Base
        ======================================================================
        From university students to corporate professionals and everyday investors—
        I combine the deep knowledge of all 17 Zerodha Varsity modules with
        Agentic AI to bring you the best of market intelligence.

        Explore the markets by asking me:
        • [Live Ticker] What is the current stock price of Tata Motors?
        • [Breaking News] Give me the latest news on the RBI rate cut.
        • [Varsity: Finance] How should I structure my retirement portfolio?
        • [Varsity: Psychology] How do I manage emotions after a big loss?
        • [Varsity: Options] When should I deploy a Bull Call Spread?
        ======================================================================
        """
    print(welcome_message)

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer, PostgresStore.from_conn_string(DB_URI) as store:
        # 3. Setup BOTH tables in your database
        checkpointer.setup()
        store.setup()

        # 4. Compile the graph with BOTH memory systems
        chatbot = build_chatbot(checkpointer, store)

        # We now track BOTH the specific conversation (thread) AND the user (user_id)
        config = {
            "configurable": {
                "thread_id": "varsity_session_2026_01",
                "user_id": "student_001",  # This links to the Long-Term Store
            }
        }

        print("⚡ Connected to Postgres (Checkpointer + Store). Ready!")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            try:
                result = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                print(f"\nAssistant: {result['messages'][-1].content}")

            except Exception as e:
                print(f"\nError: {e}")
