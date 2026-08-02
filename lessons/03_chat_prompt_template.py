from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_demo.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

def main() -> None:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )

    prompt.format(input="What is the capital of France?")
    
    chain =prompt|llm|StrOutputParser()
    value = chain.invoke({"input": "What is the capital of France?"})
    print(value)

    #m3ssagesplaceholder
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ]
    )
    chain = prompt|llm|StrOutputParser()
    value = chain.invoke({"messages": [("human","My name is John Doe")],"input": "What is my name?"})
    print(value)
    
    #Fewshot prompt template
    examples = [
    {
        "question": "2 + 2",
        "answer": "4"
    },
    {
        "question": "5 + 3",
        "answer": "8"
    }
    ]


    example_prompt = PromptTemplate.from_template(
        "Question: {question}\nAnswer: {answer}"
    )


    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Answer the math question using the examples.",
        suffix="Question: {question}\nAnswer:",
        input_variables=["question"]
    )


   

    chain = prompt | llm | StrOutputParser()


    result = chain.invoke({
        "question": "10 + 5"
    })

    print(result)

    # batch prompt template

    prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
    )
    chain = prompt | llm | StrOutputParser()

    results = chain.batch([
        {"topic": "recursion"},
        {"topic": "decorators"},
        {"topic": "generators"}
    ])

    #Json output parser
    parser = JsonOutputParser()


    prompt = PromptTemplate(
        template="""
    Answer the question.

    {format_instructions}

    Question:
    {question}
    """,

    input_variables=["question"],

    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
    )


    chain = prompt | llm | parser


    result = chain.invoke({
        "question": "What is Python?"
    })


    print(result)



if __name__ == "__main__":
    main()