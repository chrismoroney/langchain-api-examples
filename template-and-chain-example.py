from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

system_template = "You are an expert on vocabulary. I want you to return {number} of words that rhyme with the word {word}. If you can't think of a word that rhymes with {word}, make up a word! Return ONLY a comma separated list."

chat_prompt = ChatPromptTemplate.from_messages(
    ("system", system_template),
)

chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()
result = chain.invoke({"number": "10", "word": "orange"})

print(result)