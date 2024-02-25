from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

planner = (
    ChatPromptTemplate.from_template("On a scale from 1 (weakest in the game) to 100 (strongest in the game), where does the Genshin Impact character {input} fall in? Assume 50 to be the completely average in the game.")
    | chat_model
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

facts_damage = (
    ChatPromptTemplate.from_template(
        "On a scale from 1 (weakest) to 100 (strongest), how much damage does {base_response} do from abilities or skills? Return an exact score."
    )
    | chat_model
    | StrOutputParser()
)

facts_util = (
    ChatPromptTemplate.from_template(
        "On a scale from 1 (weakest) to 100 (strongest), how useful is {base_response}'s abilities or skills? Return an exact score."
    )
    | chat_model
    | StrOutputParser()
)

facts_survival = (
    ChatPromptTemplate.from_template(
        "On a scale from 1 (weakest) to 100 (strongest), how likely is {base_response}'s to survive with HP and DEF? Return an exact score."
    )
    | chat_model
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Damage score:\n{results_1}\n\nUtility Score:\n{results_2}\n\nSurvival Score:\n{results_3}\n\n"),
            ("system", "Given a character from Genshin Impact, score the character's overall strength from 1 to 100 (weakest to strongest) based on their damage, utility, and survival score. Return an analysis and exact values instead of a range of values."),
        ]
    )
    | chat_model
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": facts_damage,
        "results_2": facts_util,
        "results_3": facts_survival,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

print(chain.invoke({"input": "Nilou"}))