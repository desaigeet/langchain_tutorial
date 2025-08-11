from langchain_core.tools import tool

@tool
def multiply(x: int, y: int) -> int:
    """
    Multiply two numbers
    """
    return x*y

@tool
def addition(x: int, y: int) -> int:
    """
    Add two numbers
    """
    return x+y

class Calculator:
    def get_tools(self):
        return [multiply, addition]

toolkit = Calculator()
tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name} => {tool.description}")

