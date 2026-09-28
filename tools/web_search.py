from ddgs import DDGS


def web_search(query, max_results=5):
    """
    Searches the web via DuckDuckGo (no API key needed) and returns a
    short list of results: title, URL, and snippet for each.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        formatted.append(f"{i}. {title}\n   {href}\n   {body}")

    return "\n\n".join(formatted)


web_search_tool = {
    "name": "web_search",
    "description": (
        "Searches the public internet via DuckDuckGo for global current events, public news, "
        "and general facts. Do NOT use this for private resumes, indexed documents, or local user files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
}
