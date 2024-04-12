from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key)

prompt = """
Does pineapple belong on pizza?
"""

prompt2 = """
If 2x  = 4, 2y = 6, and 2z  = 8, then x + y + z = 15.

Evaluate if the equation is correct, and if not, figure out the correct answer.
"""

print(llm(prompt))
print()
print(llm(prompt2))
