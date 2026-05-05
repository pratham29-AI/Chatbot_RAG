"""
Tool 1: Web Search via Tavily.

Tavily is purpose-built for LLM-driven search — it returns clean, summarised
snippets rather than raw HTML, which keeps token usage low and answers precise.
Free tier provides 1 000 API calls/month, more than enough for development.

The function is decorated with @tool so LangChain can introspect its name,
description, and argument schema automatically.
"""

from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import settings

_client = TavilyClient(api_key=settings.tavily_api_key)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, recent news, or real-time facts.

    Use this tool when the user asks about topics that are unlikely to be in
    an uploaded document — e.g. today's news, current prices, live events, or
    any fact that requires up-to-date knowledge.

    Args:
        query: A concise search query string.

    Returns:
        A formatted string with the top search results (title + snippet + URL).
    """
    try:
        response = _client.search(
            query=query,
            search_depth="basic",   # "advanced" costs more API credits
            max_results=5,
            include_answer=True,    # Tavily returns a short synthesised answer
        )
    except Exception as exc:
        return f"Web search failed: {exc}"

    parts: list[str] = []

    # Tavily's synthesised answer (not always present)
    if response.get("answer"):
        parts.append(f"**Summary:** {response['answer']}\n")

    for i, result in enumerate(response.get("results", []), start=1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "").strip()
        parts.append(f"{i}. **{title}**\n   {content}\n   Source: {url}")

    if not parts:
        return "No results found for that query."

    return "\n\n".join(parts)
