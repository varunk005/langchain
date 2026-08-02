from langchain.tools import tool
from langchain_demo.llm import get_llm


# ============================================================
# LANGCHAIN TOOLS - DELIVERY FEE EXAMPLE
#
# Goal:
#
# User asks:
# "How much is delivery for 8 km?"
#
# The model can choose a Python tool
# to calculate the delivery fee.
#
# ============================================================


# ------------------------------------------------------------
# 1. CREATE A TOOL
# ------------------------------------------------------------

@tool
def calculate_delivery_fee(distance_km: float) -> float:
    """
    Calculate the delivery fee based on delivery distance in kilometers.
    """

    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    # Simple pricing rule:
    #
    # Up to 5 km  = $5
    # Above 5 km  = $5 + $1.50 per extra km

    if distance_km <= 5:
        return 5.0

    extra_distance = distance_km - 5

    return 5 + (extra_distance * 1.5)


# ------------------------------------------------------------
# 2. TEST THE TOOL DIRECTLY
# ------------------------------------------------------------

result = calculate_delivery_fee.invoke({
    "distance_km": 8
})

print("Direct tool result:")
print(result)


# ------------------------------------------------------------
# 3. INSPECT TOOL INFORMATION
# ------------------------------------------------------------

print("\nTool name:")
print(calculate_delivery_fee.name)

print("\nTool description:")
print(calculate_delivery_fee.description)

print("\nTool arguments:")
print(calculate_delivery_fee.args)


# ------------------------------------------------------------
# 4. CREATE MODEL
# ------------------------------------------------------------

llm = get_llm()


# ------------------------------------------------------------
# 5. BIND TOOL TO MODEL
#
# bind_tools() tells the model:
#
# "You are allowed to use this tool."
#
# It DOES NOT run the tool.
# ------------------------------------------------------------

llm_with_tools = llm.bind_tools([
        calculate_delivery_fee
    ])

# ------------------------------------------------------------
# 6. ASK A QUESTION
# ------------------------------------------------------------

question = "How much would delivery cost for a distance of 8 km?"


response = llm_with_tools.invoke(question)

# ------------------------------------------------------------
# 7. CHECK MODEL RESPONSE
# ------------------------------------------------------------

print("\nModel content:")
print(response.content)

print("\nTool calls:")
print(response.tool_calls)


# ------------------------------------------------------------
# The model may return something like:
#
# [
#     {
#         "name": "calculate_delivery_fee",
#         "args": {
#             "distance_km": 8
#         },
#         "id": "...",
#         "type": "tool_call"
#     }
# ]
#
#
# IMPORTANT:
#
# The model has NOT executed our Python function yet.
#
# It is only requesting:
#
# "Please run calculate_delivery_fee
# with distance_km = 8."
# ------------------------------------------------------------


# ------------------------------------------------------------
# 8. GET THE REQUESTED TOOL CALL
# ------------------------------------------------------------

tool_call = response.tool_calls[0]

print("\nSelected tool:")
print(tool_call["name"])

print("\nTool arguments:")
print(tool_call["args"])


# ------------------------------------------------------------
# 9. EXECUTE THE TOOL
# ------------------------------------------------------------

tool_result = calculate_delivery_fee.invoke(tool_call)

print("\nTool result:")
print(tool_result)


# ------------------------------------------------------------
# For 8 km:
#
# First 5 km = $5
#
# Extra distance:
#
# 8 - 5 = 3 km
#
# Extra charge:
#
# 3 × $1.50 = $4.50
#
# Total:
#
# $5 + $4.50 = $9.50
# ------------------------------------------------------------


# ------------------------------------------------------------
# 10. SEND TOOL RESULT BACK TO MODEL
# ------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": question
    },

    # Model's request to use a tool
    response,

    # Result returned by our Python tool
    tool_result
]


final_response = llm_with_tools.invoke(messages)


print("\nFinal answer:")
print(final_response.content)


# Possible answer:
#
# "The delivery fee for 8 km is $9.50."


# ============================================================
# NOTES
# ============================================================


# ------------------------------------------------------------
# WHAT IS A TOOL?
# ------------------------------------------------------------

# A LangChain tool is basically:
#
# Python function
# +
# information describing that function
#
#
# @tool converts:
#
# def calculate_delivery_fee(...):
#
# into a tool that an LLM can understand.


# ------------------------------------------------------------
# WHERE DOES TOOL INFORMATION COME FROM?
# ------------------------------------------------------------

# Function name:
#
# calculate_delivery_fee
#
# becomes the tool name.


# Function docstring:
#
# """
# Calculate the delivery fee based on delivery distance.
# """
#
# becomes the tool description.


# Type hint:
#
# distance_km: float
#
# tells LangChain/model what input is expected.


# ------------------------------------------------------------
# bind_tools()
# ------------------------------------------------------------

# model.bind_tools([tool])
#
# means:
#
# "Here are the tools available to you."


# It DOES NOT mean:
#
# "Execute the tool immediately."


# ------------------------------------------------------------
# MODEL DECIDES WHETHER TOOL IS NEEDED
# ------------------------------------------------------------

# Example:
#
# "Tell me a joke."
#
# The delivery tool is unnecessary.
#
# The model can answer directly.


# Example:
#
# "How much is delivery for 12 km?"
#
# The delivery calculator is relevant.
#
# The model may request the tool.


# ------------------------------------------------------------
# tool_calls
# ------------------------------------------------------------

# When the model wants to use a tool:
#
# response.tool_calls
#
# contains information such as:
#
# [
#     {
#         "name": "calculate_delivery_fee",
#
#         "args": {
#             "distance_km": 8
#         }
#     }
# ]


# ------------------------------------------------------------
# TOOL REQUEST VS TOOL EXECUTION
# ------------------------------------------------------------

# Very important:
#
# response.tool_calls
#
# means:
#
# MODEL REQUESTED A TOOL
#
#
# calculate_delivery_fee.invoke(...)
#
# means:
#
# PYTHON ACTUALLY EXECUTED THE TOOL


# ------------------------------------------------------------
# WHY SEND THE RESULT BACK TO THE MODEL?
# ------------------------------------------------------------

# Our Python function may return:
#
# 9.5
#
#
# But the model can turn that into:
#
# "Your delivery fee is $9.50."


# So:
#
# Tool
#     ↓
# performs calculation / gets data
#
# Model
#     ↓
# explains the result to the user


# ============================================================
# COMPLETE FLOW
# ============================================================

# User:
#
# "How much is delivery for 8 km?"
#
#             ↓
#
# Model sees available tools
#
#             ↓
#
# Model decides:
#
# calculate_delivery_fee is useful
#
#             ↓
#
# response.tool_calls
#
#             ↓
#
# {
#     "name": "calculate_delivery_fee",
#     "args": {
#         "distance_km": 8
#     }
# }
#
#             ↓
#
# Python runs:
#
# calculate_delivery_fee.invoke(...)
#
#             ↓
#
# Result:
#
# 9.5
#
#             ↓
#
# Send tool result back to model
#
#             ↓
#
# Model gives final answer:
#
# "The delivery fee is $9.50."


# ============================================================
# QUICK MEMORY NOTES
# ============================================================

# @tool
#     ↓
# Convert normal Python function into LangChain tool


# bind_tools()
#     ↓
# Tell the model which tools are available


# response.tool_calls
#     ↓
# Model requests a tool


# tool.invoke(...)
#     ↓
# Python actually runs the tool


# send ToolMessage back
#     ↓
# Model uses the result to answer naturally


# ============================================================
# ONE-LINE MEMORY TRICK
# ============================================================

# MODEL CHOOSES THE TOOL
#
# PYTHON RUNS THE TOOL
#
# MODEL EXPLAINS THE RESULT