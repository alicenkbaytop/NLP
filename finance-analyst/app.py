import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
import os
from datetime import datetime, timedelta
import yfinance as yf

# Set up Groq API Key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize Agents with Groq Model
groq_model = Groq(id="llama3-groq-70b-8192-tool-use-preview")

# Finance Data Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=False,  # Disable stock_price since we'll use yfinance directly
            company_info=True,
            stock_fundamentals=True,
            income_statements=True,
            key_financial_ratios=True,
            analyst_recommendations=True,
            company_news=True,
            technical_indicators=True,
            historical_prices=True,
        )
    ],
    instructions=[
        "Use tables to display data.",
        "Provide clear and concise summaries.",
        "Include sources where applicable.",
    ],
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    show_tool_calls=True,
    markdown=True,
)

def get_stock_analysis(stock_name: str):
    """
    Fetch and display financial data and news for a given stock.
    """
    # Show loading spinner and message
    with st.spinner("Analyzing..."):
        st.write("Analyzing... Please wait.")

        # Fetch stock price using yfinance
        with st.expander("Show Details", expanded=False):
            st.write(f"Fetching current stock price for {stock_name}...")
            try:
                stock = yf.Ticker(stock_name)
                stock_info = stock.info
                if "currentPrice" in stock_info:
                    st.write(f"Current Price: ${stock_info['currentPrice']}")
                else:
                    st.warning(f"Current price not available for {stock_name}.")
            except Exception as e:
                st.error(f"Failed to fetch stock price: {e}")

            # Fetch analyst recommendations
            st.write(f"\nFetching analyst recommendations for {stock_name}...")
            with st.spinner("Fetching analyst recommendations..."):
                analyst_response = finance_agent.run(f"Summarize analyst recommendations for {stock_name}")
                st.write(analyst_response)

            # Fetch company information
            st.write(f"\nFetching company information for {stock_name}...")
            with st.spinner("Fetching company information..."):
                company_info_response = finance_agent.run(f"Provide company information for {stock_name}")
                st.write(company_info_response)

            # Fetch key financial ratios
            st.write(f"\nFetching key financial ratios for {stock_name}...")
            with st.spinner("Fetching key financial ratios..."):
                financial_ratios_response = finance_agent.run(f"Provide key financial ratios for {stock_name}")
                st.write(financial_ratios_response)

            # Fetch the latest news
            st.write(f"\nFetching the latest news for {stock_name}...")
            with st.spinner("Fetching the latest news..."):
                news_response = finance_agent.run(f"Provide the latest news for {stock_name}")
                st.write(news_response)

            # Fetch historical prices for the last 30 days
            st.write(f"\nFetching historical prices for {stock_name} (last 30 days)...")
            try:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                historical_data = stock.history(start=start_date, end=end_date)
                if not historical_data.empty:
                    st.write(f"Historical prices for {stock_name} from {start_date} to {end_date}:")
                    st.dataframe(historical_data[["Open", "High", "Low", "Close", "Volume"]])
                else:
                    st.warning(f"No historical data found for {stock_name}.")
            except Exception as e:
                st.error(f"Failed to fetch historical prices: {e}")

        # Generate LLM recommendation
        st.write("\nGenerating LLM recommendation...")
        with st.spinner("Generating recommendation..."):
            try:
                recommendation = finance_agent.run(
                    f"Analyze the stock {stock_name} and provide a recommendation (buy/sell/hold) based on the following data: "
                    f"1. Analyst recommendations (already fetched). "
                    f"2. Company fundamentals (already fetched). "
                    f"3. Key financial ratios (already fetched). "
                    f"4. Latest news (already fetched). "
                    f"Provide a concise recommendation in one paragraph."
                )
                
                # Extract the key recommendation from the response
                if isinstance(recommendation, dict):
                    # If the response is a dictionary, extract the "content" field
                    recommendation_text = recommendation.get("content", str(recommendation))
                else:
                    # If the response is not a dictionary, convert it to a string
                    recommendation_text = str(recommendation)

                # Clean up the recommendation text
                if 'content="' in recommendation_text:
                    # Remove the `content="` prefix and any trailing metadata
                    recommendation_text = recommendation_text.split('content="')[-1].split('"')[0].strip()

                # Replace `\n\n` with actual line breaks for better readability
                recommendation_text = recommendation_text.replace("\\n\\n", "\n\n")

                # Display the recommendation in a clean and readable format
                st.markdown(f"\n\n{recommendation_text}")
                
            except Exception as e:
                st.error(f"Failed to generate LLM recommendation: {e}")

# Streamlit UI
st.title("Stock Analysis App 📈")
st.write("Enter a stock ticker symbol (e.g., NVDA, AAPL) to get financial data and analysis.")

# Input for stock ticker
stock_name = st.text_input("Stock Ticker Symbol", value="NVDA").upper()

# Button to trigger analysis
if st.button("Analyze Stock"):
    if stock_name:
        get_stock_analysis(stock_name)
    else:
        st.warning("Please enter a valid stock ticker symbol.")