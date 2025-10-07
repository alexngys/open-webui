import logging
from typing import Iterator, List, Literal, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class ValyuLoader(BaseLoader):
    """Extract web page content from URLs using Valyu Contents API.

    This is a LangChain document loader that uses Valyu's Contents API to
    retrieve content from web pages and return it as Document objects.

    Args:
        urls: URL or list of URLs to extract content from.
        api_key: The Valyu API key.
        response_length: Content length per URL: 'short' (25k), 'medium' (50k),
            'large' (100k), 'max', or custom integer.
        extract_effort: Processing effort level ('auto', 'normal', 'high').
        continue_on_failure: Whether to continue if extraction of a URL fails.
    """

    def __init__(
        self,
        urls: Union[str, List[str]],
        api_key: str,
        response_length: Literal["short", "medium", "large", "max"] = "max",
        extract_effort: Literal["auto", "normal", "high"] = "auto",
        continue_on_failure: bool = True,
    ) -> None:
        """Initialize Valyu Contents client.

        Args:
            urls: URL or list of URLs to extract content from.
            api_key: The Valyu API key.
            response_length: Content length per URL: 'short' (25k), 'medium' (50k),
                'large' (100k), or 'max'.
            extract_effort: Processing effort level ('auto', 'normal', 'high').
                'auto' automatically determines the best effort level.
            continue_on_failure: Whether to continue if extraction of a URL fails.
        """
        if not urls:
            raise ValueError("At least one URL must be provided.")

        try:
            from valyu import Valyu
        except ImportError:
            raise ImportError(
                "valyu package not found. Please install it with: pip install valyu"
            )

        self.api_key = api_key
        self.urls = urls if isinstance(urls, list) else [urls]
        self.response_length = response_length
        self.extract_effort = extract_effort
        self.continue_on_failure = continue_on_failure
        self.client = Valyu(api_key=api_key)

    def lazy_load(self) -> Iterator[Document]:
        """Extract and yield documents from the URLs using Valyu Contents API."""
        batch_size = 10
        for i in range(0, len(self.urls), batch_size):
            batch_urls = self.urls[i : i + batch_size]
            try:
                # Make the API call
                response = self.client.contents(
                    urls=batch_urls,
                    response_length=self.response_length,
                    extract_effort=self.extract_effort,
                )

                # Check if the request was successful
                if not response.success:
                    error_msg = getattr(response, "error", "Unknown error")
                    log.error(
                        f"Valyu Contents API request failed for batch {batch_urls}: {error_msg}"
                    )
                    if not self.continue_on_failure:
                        raise Exception(f"Valyu API error: {error_msg}")
                    continue

                # Process successful results
                if hasattr(response, "results") and response.results:
                    for result in response.results:
                        url = getattr(result, "url", "")
                        content = getattr(result, "content", "")

                        if not content:
                            log.warning(f"No content extracted from {url}")
                            continue

                        # Build metadata
                        metadata = {"source": url}

                        # Add any additional metadata from the result
                        if hasattr(result, "metadata") and result.metadata:
                            result_metadata = result.metadata
                            if isinstance(result_metadata, dict):
                                metadata.update(result_metadata)

                        yield Document(
                            page_content=content,
                            metadata=metadata,
                        )
                else:
                    log.warning(f"No results returned for batch {batch_urls}")

            except Exception as e:
                if self.continue_on_failure:
                    log.error(f"Error extracting content from batch {batch_urls}: {e}")
                else:
                    raise e
