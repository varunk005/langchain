from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_demo.llm import get_llm


# ============================================================
# SEQUENTIAL CHAIN EXAMPLE
#
# Goal:
# 1. Take messy user feedback
# 2. Clean the feedback
# 3. Pass cleaned feedback to another prompt
# 4. Summarize it
#
# Main idea:
#
# input
#   ↓
# prompt1
#   ↓
# model
#   ↓
# RunnableLambda
#   ↓
# prompt2
#   ↓
# model
#   ↓
# final answer
# ============================================================


# ------------------------------------------------------------
# 1. Create the model
# ------------------------------------------------------------

llm = get_llm()


# ------------------------------------------------------------
# 2. First prompt
#
# This prompt receives:
#
# {
#     "feedback": "..."
# }
#
# Its job is to clean messy feedback.
# ------------------------------------------------------------

clean_prompt = PromptTemplate.from_template(
    """
Clean and rewrite this user feedback clearly.

User feedback:
{feedback}

Cleaned feedback:
"""
)


# ------------------------------------------------------------
# 3. Second prompt
#
# This prompt expects:
#
# {
#     "cleaned_feedback": "..."
# }
#
# Its job is to summarize the cleaned feedback.
# ------------------------------------------------------------

summary_prompt = PromptTemplate.from_template(
    """
Summarize this user feedback in one short sentence.

Feedback:
{cleaned_feedback}

Summary:
"""
)


# ------------------------------------------------------------
# 4. RunnableLambda
#
# IMPORTANT:
#
# After:
#
# clean_prompt | model
#
# the model returns an AIMessage.
#
# Example:
#
# AIMessage(
#     content="The app is good, but login is slow."
# )
#
# But summary_prompt expects:
#
# {
#     "cleaned_feedback": "The app is good..."
# }
#
# So RunnableLambda converts the output into the correct shape.
# ------------------------------------------------------------

format_cleaned_output = RunnableLambda(
    lambda output: {
        "cleaned_feedback": output.content
    }
)


# ------------------------------------------------------------
# 5. Build the sequential chain
#
# Read "|" as:
#
# "send the output to the next step"
# ------------------------------------------------------------

chain = (
    clean_prompt
    | llm
    | format_cleaned_output
    | summary_prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------------------------
# 6. Give input to the chain
#
# clean_prompt expects the key:
#
# {feedback}
#
# So we send:
#
# {"feedback": "..."}
# ------------------------------------------------------------

result = chain.invoke({
    "feedback":
        "app is very good but login takes forever and sometimes it crashes"
})


# ------------------------------------------------------------
# 7. Print the final result
# ------------------------------------------------------------

print(result)


# ============================================================
# HOW THE DATA MOVES
# ============================================================

# STEP 1
#
# Input:
#
# {
#     "feedback":
#     "app is very good but login takes forever..."
# }


# STEP 2
#
# clean_prompt creates something like:
#
# Clean and rewrite this user feedback clearly.
#
# User feedback:
# app is very good but login takes forever...
#
# Cleaned feedback:


# STEP 3
#
# model returns:
#
# AIMessage(
#     content=
#     "The app is good overall, but login is slow
#      and it sometimes crashes."
# )


# STEP 4
#
# RunnableLambda changes:
#
# AIMessage(...)
#
# into:
#
# {
#     "cleaned_feedback":
#     "The app is good overall, but login is slow
#      and it sometimes crashes."
# }


# STEP 5
#
# summary_prompt now receives the correct variable:
#
# {cleaned_feedback}


# STEP 6
#
# The second model summarizes it.
#
# Possible answer:
#
# "Users like the app but experience slow logins
#  and occasional crashes."


# ============================================================
# MEMORY NOTES
# ============================================================

# PromptTemplate
# = creates a prompt with variables


# model
# = gives an AIMessage


# output.content
# = gets only the text from AIMessage


# RunnableLambda
# = custom Python function inside a LangChain chain


# StrOutputParser()
# = converts final AIMessage into a plain Python string


# Sequential chain:
#
# output of one step
# becomes
# input of the next step


# Most important rule:
#
# If the next prompt expects:
#
# {cleaned_feedback}
#
# then the previous step must give:
#
# {
#     "cleaned_feedback": "some value"
# }


# ============================================================
# SHORT VERSION TO REMEMBER
# ============================================================

# chain = (
#     prompt1
#     | model
#     | RunnableLambda(
#         lambda output: {
#             "variable_for_prompt2": output.content
#         }
#     )
#     | prompt2
#     | model
#     | StrOutputParser()
# )