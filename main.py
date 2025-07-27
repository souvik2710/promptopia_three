import asyncio
import os
import google.generativeai as genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# ---- 1. Define Toolset ----
toolset = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:8080/mcp/stream"
    )
)

# ---- 2. Define LLM Agent ----
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

# ---- 3. Define Session and Runner ----
app_name, user_id, session_id = "stock_transaction_app", "user1", "session1"
session_service = InMemorySessionService()

async def main():
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    print("Initial state:", session.state)

    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)
    user_input = Content(parts=[Part(text="Fetch my stock transactions.")])

    agent_stream = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_input)

    try:
        async for event in agent_stream:
            print("🧠 Agent Event:", event)
            if event.is_final_response():
                print("Agent Final Response:")
                print(event.content.parts[0].text)
                prompt_question = "Analyze the response and provide insights only in 200 words."
                response = model.generate_content(f"{prompt_question} {event.content.parts[0].text}")
                print("Final Response:", response.text)
    except Exception as e:
        print("⚠️ Error during agent stream:", e)
    finally:
        # Clean up async generator to prevent cancel scope error
        await agent_stream.aclose()

# Run the async main
asyncio.run(main())