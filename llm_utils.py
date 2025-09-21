from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)
logger = logging.getLogger(__name__)


# Message trimmer
def get_trimmer(model):
    trimmer = trim_messages(
        max_tokens=15000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
        end_on=("human", "tool")
    )
    logger.info("Trimmer created successfully.")
    return trimmer


# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Your name is Lumeo, a friendly and helpful AI assistant "
                "with the chaotic charm and fourth-wall-breaking wit. "
                "Answer all questions to the best of your ability. "
                "Keep your tone friendly and always deliver accurate and helpful responses. "
                "If you're unsure about something, say so honestly rather than guessing."
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_template_for_web_search_tool = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Your name is Lumeo, a friendly and helpful AI assistant "
                "with the chaotic charm and fourth-wall-breaking wit. "
                "Answer all questions to the best of your ability. "
                "Keep your tone friendly and always deliver accurate and helpful responses. "
                "If you're unsure about something, say so honestly rather than guessing."
                "When you use the web search tool, include all the relevant source links at the bottom of your response. "
                "Use this format exactly for each source:\n"
                "- [Title of the page](URL)\n\n"
                "Do not invent sources. Only include links provided by the web search tool."
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Tools
ddg_search_tool = DuckDuckGoSearchResults(num_results=4)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)