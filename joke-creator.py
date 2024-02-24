from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

template = "Tell me a joke about {subject}"

chat_prompt = ChatPromptTemplate.from_template(template)

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]

chain = (
    chat_prompt
    | chat_model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)

result = chain.invoke({"subject": "US history"})

print(result)
print()