import streamlit as st
import asyncio
import os
import google.generativeai as genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def initialize_components():
    """Initialize Google API and agent components"""
    try:
        google_api_key = os.environ.get("GEMINI_API_KEY")
        if not google_api_key:
            st.error("GEMINI_API_KEY not found in environment variables")
            return None, None, None, None
        
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Define Toolset
        toolset = MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="http://localhost:8080/mcp/stream"
            )
        )
        
        # Define LLM Agent
        root_agent = LlmAgent(
            name='fetch_stock_transactions_agent',
            model='gemini-2.0-flash',
            description=(
                'Agent for accessing all stock transactions from accounts connected to the Fi Money app. '
                'Use cases: Retrieve all stock transactions, including ISIN as identifier, transaction type (Buy, Sell, Dividend), '
                'transaction date, and NAV value of each transaction. Data is sourced directly from user-linked accounts in Fi Money.'
            ),
            instruction=(
                'Assist the user in accessing their stock transactions from accounts connected to the Fi Money app. '
                'Provide details such as ISIN, transaction type (Buy, Sell, Dividend), transaction date, and NAV value. '
                'Only use the fetch_stock_transactions tool and do not estimate or fabricate data. Return only actual data available from Fi Money.'
            ),
            tools=[toolset],
            output_key="last_result"
        )
        
        # Define Session Service
        session_service = InMemorySessionService()
        
        return model, root_agent, session_service, toolset
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

async def fetch_stock_transactions(model, agent, session_service, user_input_text):
    """Async function to fetch stock transactions"""
    app_name, user_id, session_id = "stock_transaction_app", "user1", "session1"
    
    try:
        # Create session
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # Create runner
        runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
        user_input = Content(parts=[Part(text=user_input_text)])
        
        # Run agent
        agent_stream = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)
        
        events = []
        final_response = None
        
        async for event in agent_stream:
            events.append(str(event))
            if event.is_final_response():
                final_response = event.content.parts[0].text
                break
        
        # Generate insights
        insights = None
        if final_response:
            prompt_question = "Analyze the response and provide insights only in 200 words."
            response = model.generate_content(f"{prompt_question} {final_response}")
            insights = response.text
        
        # Clean up
        await agent_stream.aclose()
        
        return final_response, insights, events
        
    except Exception as e:
        return None, None, [f"Error: {str(e)}"]

def run_async_function(coro):
    """Helper function to run async code in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def main():
    st.set_page_config(
        page_title="Stock Transaction Agent",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Transaction Agent")
    st.markdown("---")
    
    # Initialize session state
    if 'components_initialized' not in st.session_state:
        st.session_state.components_initialized = False
        st.session_state.model = None
        st.session_state.agent = None
        st.session_state.session_service = None
        st.session_state.toolset = None
    
    # Initialize components
    if not st.session_state.components_initialized:
        with st.spinner("Initializing components..."):
            model, agent, session_service, toolset = initialize_components()
            if all([model, agent, session_service, toolset]):
                st.session_state.model = model
                st.session_state.agent = agent
                st.session_state.session_service = session_service
                st.session_state.toolset = toolset
                st.session_state.components_initialized = True
                st.success("Components initialized successfully!")
                st.rerun()  # Refresh to show the main interface
            else:
                st.error("Failed to initialize components. Please check your configuration.")
                return
    
    # Only show main interface after components are initialized
    if not st.session_state.components_initialized:
        return
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration Status")
        
        # Show connection status
        if st.session_state.components_initialized:
            st.success("✅ Agent Initialized")
            st.success("✅ MCP Toolset Connected")
            st.info("🔗 MCP Server: localhost:8080")
        else:
            st.error("❌ Components Not Initialized")
        
        # Environment check
        if os.environ.get("GEMINI_API_KEY"):
            st.success("✅ Gemini API Key Found")
        else:
            st.error("❌ Gemini API Key Missing")
    
    with col2:
        st.subheader("Request Stock Transactions")
        
        # User input
        user_input = st.text_area(
            "Enter your request:",
            value="Fetch my stock transactions.",
            height=100,
            help="Enter your request to fetch stock transactions from Fi Money app"
        )
        
        # Submit button
        if st.button("🔍 Fetch Transactions", type="primary"):
            if user_input.strip():
                with st.spinner("Fetching stock transactions..."):
                    try:
                        final_response, insights, events = run_async_function(
                            fetch_stock_transactions(
                                st.session_state.model,
                                st.session_state.agent,
                                st.session_state.session_service,
                                user_input
                            )
                        )
                        
                        # Store results in session state
                        st.session_state.final_response = final_response
                        st.session_state.insights = insights
                        st.session_state.events = events
                        
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
            else:
                st.warning("Please enter a request.")
    
    # Display results
    if hasattr(st.session_state, 'final_response') and st.session_state.final_response:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Transaction Data", "💡 AI Insights"])
        
        with tab1:
            st.subheader("Stock Transaction Data")
            st.text_area(
                "Raw Response:",
                value=st.session_state.final_response,
                height=300,
                disabled=True
            )
            
            # Download button
            if st.session_state.final_response:
                st.download_button(
                    label="📥 Download Transaction Data",
                    data=st.session_state.final_response,
                    file_name=f"stock_transactions_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with tab2:
            if st.session_state.insights:
                st.subheader("AI-Generated Insights")
                st.markdown(st.session_state.insights)
            else:
                st.info("No insights available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>Stock Transaction Agent powered by Google ADK and Gemini AI</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()