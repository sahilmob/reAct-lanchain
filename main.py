from dotenv import load_dotenv
from langchain.agents import tool

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the input text by characters"""
    return len(text)


if __name__ == "__main__":
    tools = [get_text_length]
