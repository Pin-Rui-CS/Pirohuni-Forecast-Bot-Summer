from __future__ import annotations

import logging
import sys

_NOISY_LOGGERS = (
    "crawl4ai",
    "httpx",
    "httpcore",
    "hpack",
    "h2",
    "openai",
    "LiteLLM",
    "litellm",
    "streamlit",
    "asyncio",
    "playwright",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
)

# Top-level logger names that belong to this project. Only these may write
# DEBUG records (full prompts/responses) to the run log file; third-party
# libraries are limited to INFO and above so the file stays readable.
_PROJECT_LOGGER_PREFIXES = {
    "__main__",
    "Adapters",
    "Crawl4AI",
    "artifacts",
    "compiler",
    "eval_tools",
    "forecasters",
    "forecasting_bot",
    "llm_client",
    "metaculus_client",
    "monetary_cost_manager",
    "orchestrator",
    "query_maker",
    "research",
    "resolution_criteria_scraper",
    "run_logging",
    "utils",
}

_CONFIGURED = False


class _ProjectDebugOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.INFO:
            return True
        return record.name.split(".")[0] in _PROJECT_LOGGER_PREFIXES


def setup_run_logging(log_file_path: str | None = None) -> None:
    """Configure logging for a bot run.

    Console gets INFO and above in a compact format. The optional log file
    gets DEBUG from project modules (full prompts/responses live there, not
    on the console) but only INFO and above from third-party libraries.
    Known-noisy third-party loggers are capped at WARNING entirely.
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not _CONFIGURED:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)
        _CONFIGURED = True

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(_ProjectDebugOnlyFilter())
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        )
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
