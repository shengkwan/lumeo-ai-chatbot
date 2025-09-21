# 🛸 Lumeo AI Chatbot

Lumeo AI is a **GenAI-powered chatbot** built with **Ollama**, **LangChain**,
**LangGraph**, and **Streamlit**, designed with a playful personality
and real-time information retrieval.\
It supports **web search** and **conversational memory**, making it a
versatile assistant for interactive AI experiences.

------------------------------------------------------------------------

## ✨ Features

-   🤖 **Conversational AI** powered by [Ollama](https://ollama.com)
    (Qwen2.5 model by default).
-   🌐 **Web Search Integration** via **Tavily API** or **DuckDuckGo**,
    with automatic source citations.
-   🧠 **Memory & Context Management** using LangGraph checkpointing +
    configurable message trimming.
-   🎭 **Customizable Personality**: Lumeo responds with chaotic charm,
    wit, and honesty.
-   💻 **Streamlit UI** with sidebar controls, chat history, and
    response streaming.

------------------------------------------------------------------------

## 🔧 Tech Stack

-   [LangChain](https://www.langchain.com/) +
    [LangGraph](https://github.com/langchain-ai/langgraph)
-   [Ollama](https://ollama.com) (Qwen2.5 model)
-   [Streamlit](https://streamlit.io/)
-   [Tavily API](https://tavily.com) for web search

------------------------------------------------------------------------

## 🚀 Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/lumeo-genai-chatbot.git
cd lumeo-genai-chatbot
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your API keys if
needed:

``` env
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Run the Streamlit app

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📂 Project Structure

    lumeo-genai-chatbot/
    │── app.py                  # Streamlit entry point
    │── workflow.py             # LangGraph workflow builder
    │── llm_utils.py            # Prompt templates, trimmer, tools
    │── streamlit_utils.py      # Session state helpers for UI
    │── requirements.txt        # Python dependencies
    │── .env.example            # Example environment variables
    │── README.md               # Project documentation

------------------------------------------------------------------------

## ⚖️ License

This project is licensed under the **MIT License** -- see the
[LICENSE](LICENSE) file for details.

------------------------------------------------------------------------

## 🙌 Acknowledgements

-   [LangChain](https://www.langchain.com/) &
    [LangGraph](https://github.com/langchain-ai/langgraph) for
    orchestration.
-   [Streamlit](https://streamlit.io/) for the chat interface.
-   [Ollama](https://ollama.com) for running local LLMs.
-   [Tavily](https://tavily.com) and
    [DuckDuckGo](https://duckduckgo.com) for web search integration.

------------------------------------------------------------------------
