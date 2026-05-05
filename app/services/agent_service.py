"""
Agent Service — LangChain OpenAI-tools agent wired with all three tools.

Architecture
────────────
  User message
       │
       ▼
  AgentExecutor  ◄──────────── system prompt + chat history
       │
       ├─► search_documents  (FAISS RAG)
       ├─► web_search        (Tavily)
       └─► calculator        (safe AST eval)
       │
       ▼
  Final streaming response

The agent uses OpenAI's native function/tool-calling capability
(`create_openai_tools_agent`) which lets the model decide autonomously when
to invoke a tool, chain multiple tool calls, or respond directly.

Streaming
─────────
`AgentExecutor.astream_events(version="v2")` emits fine-grained events.
We filter for `on_chat_model_stream` and yield only chunks whose `.content`
is non-empty text (skipping internal tool-call tokens, which carry content
in `.tool_call_chunks` and have an empty `.content` string).
"""

import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from app.config import settings
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.tools.calculator_tool import calculator
from app.tools.web_search_tool import web_search

# ── RAG tool (defined here so it closes over rag_service) ────────────────────

def _make_search_documents_tool():
    from app.services.rag_service import rag_service

    @tool
    def search_documents(query: str) -> str:
        """
        Search the uploaded documents for information relevant to the query.

        Always call this tool FIRST when the user might be asking about content
        in an uploaded PDF or text file.  Returns relevant excerpts with source
        and page information, or a clear message when nothing matches.

        Args:
            query: The question or topic to search for in the documents.

        Returns:
            Formatted document excerpts, or a message explaining why no results
            were found.
        """
        if not rag_service.has_documents():
            return (
                "No documents have been uploaded yet.  "
                "Ask the user to upload a PDF or text file via POST /documents/upload, "
                "or offer to answer using web search instead."
            )

        results = rag_service.search(query)

        if not results:
            return (
                "No relevant information found in the uploaded documents for that query.  "
                "Do not guess — tell the user the information is not available in the documents."
            )

        lines = ["Here are the most relevant excerpts from the uploaded documents:\n"]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"[{i}] Source: {r['source']} | Page: {r['page']} "
                f"| Relevance: {r['score']:.0%}\n"
                f"{r['content']}\n"
            )
        return "\n".join(lines)

    return search_documents


# ── agent factory ─────────────────────────────────────────────────────────────

def build_agent_executor() -> AgentExecutor:
    """
    Construct a fresh AgentExecutor.  Called once at startup; the returned
    executor is stateless — conversation history is injected per request.
    """
    llm = ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openai_api_key,
        temperature=0,          # deterministic; set higher for creative tasks
        streaming=True,         # required for token-by-token streaming
    )

    search_documents = _make_search_documents_tool()
    tools = [search_documents, web_search, calculator]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),   # injected per request
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # agent's internal reasoning
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,          # set True to log tool calls to stdout
        max_iterations=6,       # prevent runaway loops
        handle_parsing_errors=True,
    )


# ── streaming helper ──────────────────────────────────────────────────────────

async def stream_agent_response(
    executor: AgentExecutor,
    user_message: str,
    chat_history: list[BaseMessage],
):
    """
    Async generator that yields plain-text tokens from the agent's final answer.

    Token filtering:
      - `on_chat_model_stream` fires for every LLM call, including intermediate
        ones where the model decides which tool to call.  Those chunks have
        empty `.content` and populated `.tool_call_chunks`.
      - We yield only chunks where `content` is a non-empty string, which
        corresponds to the final human-readable answer.

    Yields:
        str — individual text tokens as they are produced.
    """
    collected: list[str] = []

    async for event in executor.astream_events(
        {"input": user_message, "chat_history": chat_history},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            text = chunk.content
            if isinstance(text, str) and text:
                collected.append(text)
                yield text

    full_response = "".join(collected)
    yield json.dumps({"__final__": full_response})
