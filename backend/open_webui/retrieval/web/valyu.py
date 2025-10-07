import logging
import json
from typing import Optional

from open_webui.retrieval.web.main import SearchResult, get_filtered_results
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def search_valyu(
    api_key: str,
    query: str,
    count: int,
    filter_list: Optional[list[str]] = None,
) -> list[SearchResult]:
    """Search using Valyu's Search API and return the results as a list of SearchResult objects.

    Args:
        api_key (str): A Valyu Search API key
        query (str): The query to search for
        count (int): The maximum number of results to return
        filter_list (Optional[list[str]]): Optional list of domains to filter results by

    Returns:
        list[SearchResult]: A list of search results
    """
    try:
        from valyu import Valyu
    except ImportError:
        log.error("valyu package not found. Please install it with: pip install valyu")
        return []

    log.info(f"Searching with Valyu for query: {query}")

    try:
        # Initialize Valyu client
        client = Valyu(api_key=api_key)

        # Perform search
        response = client.search(
            query=query,
            max_num_results=count or 10,
        )

        # Check if the request was successful
        if not response.success:
            error_msg = getattr(response, "error", "Unknown error")
            log.error(f"Valyu Search API request failed: {error_msg}")
            return []

        # Extract results
        results = []
        if hasattr(response, "results") and response.results:
            for result in response.results:
                url = getattr(result, "url", "")
                title = getattr(result, "title", "")
                content = (
                    getattr(result, "content", "")
                    or getattr(result, "text", "")
                    or getattr(result, "snippet", "")
                )
                data_type = getattr(result, "data_type", "unstructured")

                # Convert structured data to string if needed
                if data_type == "structured" and content:
                    if isinstance(content, dict) or isinstance(content, list):
                        try:
                            content = json.dumps(content, indent=2)
                        except Exception as e:
                            log.warning(
                                f"Failed to convert structured content to JSON string: {e}"
                            )
                            content = str(content)
                    elif not isinstance(content, str):
                        content = str(content)

                if url:
                    results.append(
                        {
                            "url": url,
                            "title": title,
                            "content": content,
                            "data_type": data_type,
                        }
                    )

        # Apply domain filtering if provided
        if filter_list and results:
            results = get_filtered_results(results, filter_list)

        log.info(f"Found {len(results)} results")

        return [
            SearchResult(
                link=result["url"],
                title=result.get("title", ""),
                snippet=result.get("content", ""),
            )
            for result in results
        ]

    except Exception as e:
        log.error(f"Error searching with Valyu: {e}")
        return []
