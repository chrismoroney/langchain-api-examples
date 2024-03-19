from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(openai_api_key=api_key)

template = """
    write some code to solve the user's problem.

    Return only python code in Markdown format, e.g.:
    ```python
    ....
    ```
"""

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run

result = chain.invoke({"input": "Give me a random number between 1 and 100"})

print(result)