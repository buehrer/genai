from typing import Annotated, Literal
import os
from dotenv import load_dotenv
from autogen import ConversableAgent
import openai
from openai import OpenAIError
import promptflow
from promptflow.tracing import trace as trace_nabila, start_trace

# Instrument traces
start_trace()


# Load environment variables from the .env file
load_dotenv()

# Get the API key and other configurations from the environment variables
api_key = os.getenv("GPT4O_API_KEY")
model = os.getenv("GPT4O_MODEL")
base_url = os.getenv("GPT4O_BASE")
api_version = os.getenv("GPT4O_VERSION")
api_type = os.getenv("OPENAI_API_TYPE")


if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

if not api_version:
    raise ValueError("GPT4O_API_VERSION environment variable is not set")

# Define the type of the operator
Operator = Literal["+", "-", "*", "/"]


# Define the calculator function 
@trace_nabila
def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")



# Let's first define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple calculations. "
    "Return 'TERMINATE' when the task is done.",
    llm_config={
        "config_list": [
            {
                "model": model,  # Use the verified model name
                "api_type": api_type,  # Use the verified API type
                "api_key": api_key,  # Use the verified API key
                "base_url": base_url,  # Use the verified base URL
                "api_version": api_version  # Use the verified API version
            }
        ]
    },
)


# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)


# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="calculator")(calculator)


try:
    chat_result = user_proxy.initiate_chat(assistant, message="What is (44232 + 13312 / (232 - 32)) * 5?")
    chat_result = user_proxy.initiate_chat(assistant, message="What is 9*3+3")

except OpenAIError as e:
    print(f"An error occurred: {e}")