from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    x: int = Field(required=True, description="First Number to Multiply")
    y: int = Field(required=True, description="Second Number to Multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self, x: int, y: int) -> int:
        return x*y

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"x":2, "y":3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
