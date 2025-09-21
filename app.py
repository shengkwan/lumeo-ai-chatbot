from langchain_ollama import ChatOllama
import streamlit as st
import time
from workflow import LLMWorkflow
from langchain_core.messages import HumanMessage
import uuid
from llm_utils import get_trimmer
from streamlit_utils import initialise_session_state, disable_chat_input
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup streamlit page config
st.set_page_config(
    page_title="Lumeo AI",
    page_icon=":space_invader:",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Setup streamlit page title block
st.html(r"title_block\title_block_2.html")

with st.sidebar:
    # Setup web search toggle
    use_web_search = st.toggle(
        ":material/travel_explore: Web search", 
        value=False, 
        help=(
            "Enable this to allow Lumeo to use the Tavily Search API "
            "for real-time web search results, based on your query.\n\n"
            "Note: Turning this on will consume the Tavily API credits."
        )
    )

    # Setup clear chat and memory button
    if st.button("Clear", icon=":material/cleaning_services:", help="Clear chat & memory"):
        st.session_state.messages = []
        logger.info("Cleared chat history.")
        
        st.session_state.thread_id = uuid.uuid4()
        logger.info("Thread ID has been reset.")
        
        st.rerun()

# Initialise session state
thread_id, workflow = initialise_session_state()


# Accept user input
if prompt := st.chat_input(
    placeholder="What is up?", 
    accept_file=True, 
    file_type="pdf", 
    max_chars=5000, 
    disabled=st.session_state.is_generating, 
    on_submit=disable_chat_input
):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "avatar": ":material/taunt:", 
        "content": prompt.text
    })
    
    # Display user message in chat message container
    with st.chat_message("user", avatar=":material/taunt:"):
        st.markdown(prompt.text)
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "use_web_search": use_web_search
        }
    }

    assistant_placeholder = st.empty()

    try:
        stream = workflow.stream(
            {"messages": [HumanMessage(prompt.text)]},
            config,
            stream_mode="messages"
        )
        
        # Create stream generator function for st.write_stream purpose 
        def stream_generator():
            full_response = ""
            # try:
            for chunk, metadata in stream:
                if chunk.content and metadata["langgraph_node"] == "llm":
                    full_response += chunk.content
                    yield chunk.content
            time.sleep(0.1)
            st.session_state.messages.append({
                "role": "assistant", 
                "avatar": ":material/network_intelligence:", 
                "content": full_response
            })
            logger.info("AI response successfully streamed and saved to memory.")
        
        with assistant_placeholder.container():
            with st.chat_message("assistant", avatar=":material/network_intelligence:"):
                with st.spinner("Generating response...", show_time=True):
                    try:
                        st.write_stream(stream_generator())
                    except Exception as e:
                        logger.exception(f"Error while streaming response: {str(e)}")
                        error_message = f"**Error while streaming response:**  \n{str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "avatar": ":material/network_intelligence:",
                            "type": "error",
                            "content": error_message
                        })

    except Exception as e:
        logger.exception(f"Workflow stream failed: {str(e)}")
        error_message = f"**Workflow stream failed:**  \n{str(e)}"
        st.session_state.messages.append({
            "role": "assistant", 
            "avatar": ":material/network_intelligence:",
            "type": "error",
            "content": error_message
        })
        with st.chat_message("assistant", avatar=":material/network_intelligence:"):
            st.error(error_message)
    finally:
        st.session_state.is_generating = False
        st.rerun()