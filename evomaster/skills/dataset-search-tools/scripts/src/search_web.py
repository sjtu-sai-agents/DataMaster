import httpx
import os
import http.client
import json


def google_search(query: str) -> str:
    """
    [General Web Search via Google] Perform a broad, general web search (Google Search) for any topic in any language.

    This is the **primary search tool** and should be used first to identify relevant web pages.
    It returns a structured JSON object containing snippets (summaries), titles, and crucially, the **URLs (web links)** of matching results.

    **AI Usage Guideline:**
    1.  Use this function to find the relevant URL(s) for a given query.
    2.  Once you have a specific URL of interest, you **must** pass that URL to the `web_parse` function to retrieve the full content of that page for detailed analysis.

    Args:
        query: The search query, which can be in any language (English, Chinese, etc.).

    Returns:
        A JSON string containing the search results, including snippets, titles, and the essential web links (URLs).
    """
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    conn.request("POST", "/search", payload, headers)
    data = conn.getresponse().read().decode("utf-8")
    return data


def web_parse(url: str) -> str:
    """
    [Specific Web Page Content Extractor] Fetch and extract the full, clean text content from a specific web page given its URL.

    This tool is designed for deep content retrieval. It takes a complete URL and returns the entire, main body content of that page, stripped of irrelevant elements like headers, footers, and advertisements.

    **AI Usage Guideline (Recommended Workflow):**
    1.  **DO NOT** use this function for general searching.
    2.  First, call `Google Search` with your keywords to get a list of potential URLs.
    3.  Then, call `web_parse` using a specific URL retrieved from the `Google Search` output to get the complete text for summary or detailed fact-checking.

    Args:
        url: The complete, absolute URL of the page to scrape (e.g., 'https://www.example.com/article-title').

    Returns:
        A JSON string containing the full, readable content of the specified URL.
    """
    conn = http.client.HTTPSConnection("scrape.serper.dev")
    payload = json.dumps({"url": url, "includeMarkdown": True})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    conn.request("POST", "/", payload, headers)
    data = conn.getresponse().read().decode("utf-8")
    return data
