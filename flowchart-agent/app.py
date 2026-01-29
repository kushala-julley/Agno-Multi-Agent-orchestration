import streamlit as st
import time
from typing import Optional, Iterator
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os
import re
from streamlit_mermaid import st_mermaid
import uuid

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model configurations
MODEL_CONFIGS = {
    "Ollama (Local)": {
        "models": [
            "llama3.2",
            "llama3.1",
            "llama3",
            "qwen2.5",
            "deepseek-r1",
        ],
        "host": OLLAMA_HOST,
        "model_class": Ollama,
        "requires_api_key": False
    }
}

# Page config
st.set_page_config(
    page_title="AI Flowchart Generator",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_diagram" not in st.session_state:
    st.session_state.current_diagram = None
if "diagram_explanation" not in st.session_state:
    st.session_state.diagram_explanation = ""
if "diagram_key" not in st.session_state:
    st.session_state.diagram_key = str(uuid.uuid4())
if "llm_provider" not in st.session_state:
    # Default to local so the app can run without API keys.
    st.session_state.llm_provider = "Ollama (Local)"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_CONFIGS[st.session_state.llm_provider]["models"][0]
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = OLLAMA_HOST

# Sidebar
with st.sidebar:
    st.title("AI Agent Settings")
    
    # Only local Ollama is offered
    provider = "Ollama (Local)"
    st.session_state.llm_provider = provider
    st.text_input(
        "Ollama Host",
        value=st.session_state.ollama_host,
        help="Default is http://localhost:11434. If running Ollama elsewhere, set OLLAMA_HOST or edit this.",
        key="ollama_host",
    )

    # Update available models based on provider
    available_models = MODEL_CONFIGS[provider]["models"]
    model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        key="selected_model"
    )
    
    # Agent selection
    agent = st.selectbox(
        "Select AI Agent",
        ["Mermaid Flowchart Agent"],
        index=0
    )
    
    # Diagram Settings
    st.subheader("Diagram Settings")
    diagram_height = st.slider("Diagram Height", 200, 800, 600, 50)
    show_controls = st.checkbox("Show Diagram Controls", value=True)
    
    # Provider status
    st.subheader("Provider Status")
    st.info("Using local Ollama (no API key needed).")

# Initialize agent with selected model
selected_config = MODEL_CONFIGS[st.session_state.llm_provider]
model_instance = selected_config["model_class"](
    id=st.session_state.selected_model,
    host=st.session_state.ollama_host,
)

mermaid_agent = Agent(
    name="Mermaid Flowchart Agent",
    role="get mermaid diagram information",
    model=model_instance,
    tools=[DuckDuckGoTools()],
    instructions="""
    You are a professional Senior God level full stack Developer. Designed countless SaaS application products and have 1000+ hours of experience in the field.
    You are also an expert in the field of flowchart and diagram design.
    Your current role is to design flowcharts and diagrams for applications based on user requests.
    The designs you create should be professional and modern.
    You must respond with a Mermaid diagram in the following format:
    
    ```mermaid
    [Your Mermaid diagram code here]
    ```
    
    Follow these guidelines:
    1. Use appropriate Mermaid syntax (flowchart, sequence diagram, class diagram, etc.)
    2. Include clear node labels and connections
    3. Use proper indentation and formatting
    4. Add comments to explain complex parts
    5. Ensure the diagram is readable and well-structured
    
    After the diagram, provide a brief explanation of the design.
    """,
    markdown=True,
)

# Main content area
st.title("AI Flowchart Generator")

# Chat input
user_input = st.text_area(
    "Enter your flowchart request",
    placeholder="Describe the flowchart or diagram you want to create...",
    height=100
)

# Generate button
if st.button("Generate Flowchart", type="primary"):
    if user_input:
        with st.spinner("Generating flowchart..."):
            # Clear previous content
            st.session_state.messages = []
            st.session_state.current_diagram = None
            st.session_state.diagram_key = str(uuid.uuid4())
            st.session_state.diagram_explanation = ""
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            response_stream: Iterator = mermaid_agent.run(user_input, stream=True)
            full_response = ""
            
            # First collect the complete response
            for response in response_stream:
                if response.content:
                    full_response += response.content
            
            # Extract diagram and explanation
            mermaid_match = re.search(r'```mermaid\n(.*?)\n```', full_response, re.DOTALL)
            if mermaid_match:
                diagram_code = mermaid_match.group(1)
                st.session_state.current_diagram = diagram_code
                
                # Display the diagram
                st.subheader("Generated Diagram")
                st_mermaid(
                    diagram_code,
                    height=diagram_height,
                    show_controls=show_controls,
                    key=f"mermaid_{st.session_state.diagram_key}"
                )
            
            # Extract and display explanation
            explanation = re.sub(r'```mermaid\n.*?\n```', '', full_response, flags=re.DOTALL).strip()
            if explanation:
                st.session_state.diagram_explanation = explanation
                st.subheader("Diagram Explanation")
                st.markdown(explanation)
    else:
        st.warning("Please enter a flowchart request first!")
