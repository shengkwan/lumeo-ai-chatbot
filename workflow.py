from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from llm_utils import (prompt_template, prompt_template_for_web_search_tool, ddg_search_tool, tavily_search_tool) 
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def route_tools(
    state: State,
    config: dict
):
    """
    Determines whether to route the workflow to the tools node or the LLM node.

    If web search is disabled via the config, it always routes to the LLM node.
    If web search is enabled, it checks the last AI message for tool calls:
    - If tool calls are present, routes to the tools node.
    - If no tool calls are found, routes to the LLM node.
    """
    use_web_search = config.get("configurable", {}).get("use_web_search", False)

    # Always go to LLM node if web search is off
    if not use_web_search:
        return "llm"
    
    # Otherwise check if there are tool calls
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "llm"


class LLMWorkflow:
    """
    Builds and manages a LangGraph workflow for processing conversational state 
    using a ChatOllama language model.

    Args:
        llm (ChatOllama): 
            An initialized ChatOllama model used to generate responses 
            during graph execution.
        
        trimmer:
            An initialized trimmer used to trim older message during graph execution.

    Attributes:
        llm (ChatOllama): The LLM used in the workflow.
        llm_with_tools: LLM binded with tools
        workflow: The compiled LangGraph workflow with memory checkpointing.
        trimmer: The message trimmer used in the workflow. 

    Methods:
        get_workflow(): 
            Returns the compiled LangGraph workflow, ready for execution.
    """
    def __init__(self, llm: ChatOllama, trimmer):
        self.llm = llm
        self.llm_with_tools = self._build_llm_with_tools()
        self.trimmer = trimmer
        self.workflow = self._build_workflow()

    def _build_llm_with_tools(self):
        llm_with_tools = self.llm.bind_tools([tavily_search_tool])
        return llm_with_tools

    def _call_llm_with_tools(self, state: State):
        """
        Node function that invokes the LLM binded with tools on the trimmed messages from the state with a prompt 
        and returns the tool calling argument.

        Args:
            state (State): The current state including messages.

        Returns:
            dict: A dictionary with the model's response message wrapped in a list.
        """
        try:
            logger.info("Trimming messages....")
            trimmed_messages = self.trimmer.invoke(state["messages"])
            # logger.info(f"Trimmed message: {trimmed_messages}")

            logger.info("Generating prompt from trimmed messages...")
            prompt = prompt_template.invoke(
                {"messages": trimmed_messages}
            )
            # logger.info(f"Prompt for model with tools: {prompt}")
            
            logger.info("Invoking model with tools...")
            response = self.llm_with_tools.invoke(prompt)
            if hasattr(response, "tool_calls") and len(response.tool_calls) <= 0:
                logger.info("No tool calls from model response.")
                return None
            logger.info(f"Model with tools successfully returned a response with tool calls: {response.tool_calls}")
            return {"messages": [response]}
        
        except Exception as e:
            logger.exception(f"Failed during model invocation in _call_model: {str(e)}")
            raise  # Re-raise to let LangGraph handle or fail explicitly
        
    def _call_llm(self, state: State):
        """
        Node function that invokes the LLM on the trimmed messages from the state with a prompt 
        and returns the AI response.

        Args:
            state (State): The current state including messages.

        Returns:
            dict: A dictionary with the model's response message wrapped in a list.
        """
        try:
            # logger.info(f"{state["messages"]}")
            logger.info("Trimming messages....")
            trimmed_messages = self.trimmer.invoke(state["messages"])
            logger.info(f"Trimmed message: {trimmed_messages}")

            logger.info("Generating prompt from trimmed messages...")
            # Check if the previous message is a ToolMessage
            last_msg = trimmed_messages[-1]
            if last_msg.type == "tool":
                logger.info("Detected tool message. Using prompt_template_for_web_search_tool...")
                prompt = prompt_template_for_web_search_tool.invoke(
                    {"messages": trimmed_messages}
                )
            else:
                logger.info("Using default prompt_template...")
                prompt = prompt_template.invoke(
                    {"messages": trimmed_messages}
                )
            logger.info(f"Prompt: {prompt}")
            
            logger.info("Invoking model...")
            response = self.llm.invoke(prompt)
            logger.info(f"Model successfully returned a response: {response.content}")
            return {"messages": [response]}
        
        except Exception as e:
            logger.exception(f"Failed during model invocation in _call_model: {str(e)}")
            raise  # Re-raise to let LangGraph handle or fail explicitly
    
    def _build_workflow(self):
        """
        Builds and compiles the LangGraph workflow with a single model node 
        and a memory checkpointer.

        Returns:
            workflow: A compiled LangGraph workflow object.
        """
        logger.info("Building LangGraph workflow...")
        graph = StateGraph(state_schema=State)
        tool_node = ToolNode(tools=[tavily_search_tool])

        graph.add_node("llm", self._call_llm)
        graph.add_node("llm_with_tools", self._call_llm_with_tools)
        graph.add_node("tools", tool_node)

        graph.add_conditional_edges(
            "llm_with_tools",
            route_tools
        )
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm_with_tools")
        
        memory = MemorySaver()
        workflow = graph.compile(checkpointer=memory)
        logger.info("Workflow compiled successfully.")
        
        return workflow
    
    def get_workflow(self):
        """
        Returns the compiled LangGraph workflow object.
        """
        return self.workflow
