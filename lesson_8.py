
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


# STEP 1: Define the Python tool

@tool
def calculate_electricity_cost(
    monthly_units: float,
    price_per_unit: float
) -> float:
    """Calculate monthly electricity cost in rupees.

    Args:
        monthly_units: Monthly electricity consumption in kWh.
        price_per_unit: Electricity price in rupees per kWh.
    """

    if monthly_units < 0:
        raise ValueError("Monthly units cannot be negative.")

    if price_per_unit < 0:
        raise ValueError("Price per unit cannot be negative.")

    return monthly_units * price_per_unit


# STEP 2: Initialize the model

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ),
    temperature=0.2
)


# STEP 3: Make the tool available to the model

tools = [calculate_electricity_cost]

model_with_tools = model.bind_tools(tools)

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# STEP 4: Start the conversation

question = (
    "Calculate my monthly electricity bill "
    "for 2500 units at 9 rupees per unit."
)

messages = [
    HumanMessage(content=question)
]


# STEP 5: Ask the model

response = model_with_tools.invoke(messages)


# STEP 6: Execute tool calls until the model finishes

iteration = 0
max_iterations = 5

while response.tool_calls and iteration < max_iterations:

    iteration += 1

    # Add the model's tool request to conversation history
    messages.append(response)

    # Execute each requested tool
    for tool_call in response.tool_calls:

        selected_tool = tools_by_name[tool_call["name"]]

        tool_result = selected_tool.invoke(
            tool_call["args"]
        )

        print(
            f"Executed {tool_call['name']}: {tool_result}"
        )

        # Send the result back with its corresponding call ID
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    # Ask the model to continue using the tool results
    response = model_with_tools.invoke(messages)


# STEP 7: Display the final answer

if response.tool_calls:
    print("Stopped: maximum tool-call iterations reached.")
else:
    print("\nFinal answer:")
    print(response.content)
