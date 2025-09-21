from langchain_ollama import ChatOllama
import streamlit as st
import time
from workflow import LLMWorkflow
from langchain_core.messages import HumanMessage
import uuid
from llm_utils import get_trimmer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)
logger = logging.getLogger(__name__)

def initialise_session_state():
    """
    Initialise session state variables from streamlit
    """
    # Initialize model
    if "model" not in st.session_state:
        st.session_state.model = ChatOllama(
            model = "qwen2.5:3b-instruct", 
            temperature = 0.8,
        )
        logger.info("Initialized model.")
    model = st.session_state.model

    # Initialize trimmer
    if "trimmer" not in st.session_state:
        st.session_state.trimmer = get_trimmer(model)
        logger.info("Initialized message trimmer.")
    trimmer = st.session_state.trimmer

    # Initialize model
    if "workflow" not in st.session_state:
        st.session_state.workflow = LLMWorkflow(model, trimmer).get_workflow()
        logger.info("Initialized workflow.")
    workflow = st.session_state.workflow

    # Initialize thread_id
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4()
        logger.info("Initialized thread ID.")
    thread_id = st.session_state.thread_id

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized chat history.")

    # Initialize the input control flag
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
        logger.info("Initialized chat input control flag.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if message.get("type") == "error":
                st.error(message["content"])
            else:
                st.markdown(message["content"])
    
    return thread_id, workflow


def disable_chat_input():
    """
    Disable chat input from streamlit
    """
    st.session_state.is_generating = True