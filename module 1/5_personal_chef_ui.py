import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """
You are a personal chef. The user will give you a list of ingredients they have left over their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return reciepe suggestion and eventualy the recipe instructions to the user, if requested.

"""

def query(query: str):

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    agent = create_agent(
        model = model,
        tools=[web_search],
        system_prompt = system_prompt,
        checkpointer = InMemorySaver()
    )

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config
    )

    return response["messages"][-1].text

user_message = st.chat_input("Savol bering!")
if user_message:
    st.chat_message("human").write(user_message)
    ai_message = query(user_message)
    st.chat_message("ai").write(ai_message)


