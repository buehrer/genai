import os

from promptflow import tool
from promptflow.connections import CustomConnection

from solver import run_it


@tool
def run_it_tool(
    connection: CustomConnection) -> str:

    # set environment variables
    for key, value in dict(connection).items():
        os.environ[key] = value

    # call the entry function
    return run_it(
    )