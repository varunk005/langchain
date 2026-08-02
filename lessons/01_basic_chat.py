"""
01 — Basic chat
Invoke the LLM with system + human messages.
Run: uv run lessons/01_basic_chat.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_demo.llm import get_llm


def main() -> None:
    llm = get_llm()

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain LangChain in one short paragraph."),
    ]

    response = llm.invoke(messages)
    #know about the response
    print(response.content_blocks)

    #know model details
    print(llm.profile.get("max_input_tokens"))


if __name__ == "__main__":
    main()
