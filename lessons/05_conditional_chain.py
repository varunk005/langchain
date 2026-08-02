from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_demo.llm import get_llm

# ============================================================
# MODEL
# ============================================================

llm = get_llm()


# ============================================================
# 1. POSITIVE REPLY PROMPT
# ============================================================

positive_prompt = PromptTemplate.from_template(
    """
The user gave positive feedback:

{feedback}

Thank them and politely ask them to leave a rating.
"""
)

positive_chain = (
    positive_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 2. NEUTRAL REPLY PROMPT
# ============================================================

neutral_prompt = PromptTemplate.from_template(
    """
The user gave neutral feedback:

{feedback}

Ask them politely for more details.
"""
)

neutral_chain = (
    neutral_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 3. NEGATIVE REPLY PROMPT
# ============================================================

negative_prompt = PromptTemplate.from_template(
    """
The user gave negative feedback:

{feedback}

Apologize and tell them the issue will be forwarded
to the relevant team.
"""
)

negative_chain = (
    negative_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 4. ROUTING FUNCTION
#
# This function decides WHICH chain should run.
#
# It receives a dictionary like:
#
# {
#     "sentiment": "positive",
#     "feedback": "I love this app!"
# }
# ============================================================

def route_feedback(data):

    sentiment = data["sentiment"]

    if sentiment == "positive":
        return positive_chain.invoke({
            "feedback": data["feedback"]
        })

    elif sentiment == "neutral":
        return neutral_chain.invoke({
            "feedback": data["feedback"]
        })

    else:
        return negative_chain.invoke({
            "feedback": data["feedback"]
        })


# ============================================================
# 5. TURN THE PYTHON FUNCTION INTO A RUNNABLE
# ============================================================

router = RunnableLambda(route_feedback)


# ============================================================
# 6. TEST IT
# ============================================================

result = router.invoke({
    "sentiment": "positive",
    "feedback": "I really love this app!"
})

print(result)