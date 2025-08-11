from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    x: int = Field(required=True, description="First Number to Multiply")
    y: int = Field(required=True, description="Second Number to Multiply")


def multiply(x: int, y: int) -> int:
    """
    Multtiply two number
    """
    return x * y

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    desccription = "Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"x":2, "y":3})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
