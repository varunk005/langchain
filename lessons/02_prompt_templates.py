"""
02 — Prompt templates
Fill a reusable prompt template, then send it to the LLM.
Run: uv run lessons/02_prompt_templates.py
"""

from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_demo.llm import get_llm



def main() -> None:
    llm = get_llm()
     
    #create a prompt template
    prompt = PromptTemplate.from_template(
    "Explain {topic} in very simple language.")

    result = prompt.format(topic="recursion")
    print(result)
    
    #create a prompt template with multiple variables
    prompt1 = PromptTemplate.from_template(
    """
    Teach me {topic}.
    My current level: {level}
    Use language: {language}
    Give {examples} examples.
    """
    )

    result1 = prompt1.format(
        topic="Python decorators",
        level="beginner",
        language="simple English",
        examples=3
    )

    print(result1)

    #format vs invoke
    
    prompt2 = PromptTemplate.from_template(
    "Tell me about {topic}"
    )

    text = prompt2.format(topic="black holes")

    print(type(text))
    print(text)

    value = prompt2.invoke(
    {"topic": "black holes"}
    )

    print(type(value))
    print(value)


    #.format() → "Give me the final string"
    #.invoke() → "Use this as a LangChain component"
#################################################################################################

#Prompt → Model → Parser

    prompt4 = PromptTemplate.from_template(
            """
        You are an excellent teacher.

        Explain {topic} to a {level} student.
        Use a simple example.
        """)

    model = llm

    parser = StrOutputParser()

    chain = prompt4 | model | parser

    #answer4 = chain.invoke({
    #        "topic": "recursion",
    #        "level": "beginner"
    #    })

    #print(answer4)

#Python f-string:
#data + formatting → string

#LangChain PromptTemplate:
#reusable prompt specification → component in an AI application

#################################################################################################

#with Json this is the correct way to return json with double brackets
    prompt = PromptTemplate.from_template(
    """
    Answer {question}.

    Return this structure:
    {{
        "answer": "...",
        "confidence": 0.0
    }}
    """
    )

    print(
        prompt.format(
            question="What is Python?"
        )
    )


############################################################################
     #Dynamic Prompt Templates
  
    prompt = PromptTemplate.from_template(
        """
    Today's date: {today}

    Question:
    {question}
    """
    )

    prompt = prompt.partial(
        today=lambda: datetime.now().date().isoformat()
    )

    print(
        prompt.format(
            question="Create my daily plan."
        )
    ) 


if __name__ == "__main__":
    main()
