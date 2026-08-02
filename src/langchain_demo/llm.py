"""Shared LLM setup — import get_llm() from every lesson."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


def get_llm(temperature: float = 0.7) -> ChatAnthropic:
    """Return a configured Claude Haiku chat model."""
    load_dotenv()
    return ChatAnthropic(model="claude-haiku-4-5", temperature=temperature)
