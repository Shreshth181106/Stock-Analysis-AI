# COMPLETE AI-BASED STOCK SUGGESTION TOOLs

# ---------- Install the below libraries on your system ----------
# pip install yfinance scikit-learn pandas tabulate requests textblob prettytable fuzzywuzzy python-Levenshtein

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
import numpy as np
import requests
import time
import re # Import regex for ticker validation
from textblob import TextBlob # For sentiment analysis, from news.py
from prettytable import PrettyTable # For tabular display, from news.py
from fuzzywuzzy import process # For fuzzy matching, from news.py

NEWS_API_KEY = "43b50968d7074d939ad8163eb262608b" # Your NewsAPI.org key
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything" # Base URL for NewsAPI.org

# Global variable to store stock mapping for fuzzy search (loaded once)
global_stock_mapping = {}

# ---------- STEP 1: PREPARE STOCK DATA (for Stock Suggestion Tool) ----------

def get_stock_data(stock_list):
    """
    Fetches historical stock data for a given list of stock tickers.
    Calculates average price, volatility, average volume, and assigns a risk category.
    """
    data_rows = []

    for stock in stock_list:
        try:
            ticker = yf.Ticker(stock)
            # Fetch 1 month of daily data for volatility calculation
            data = ticker.history(period="1mo", interval="1d")

            if data.empty or 'Close' not in data.columns:
                continue

            prices = pd.to_numeric(data['Close'], errors='coerce').dropna().tolist()
            volumes = pd.to_numeric(data['Volume'], errors='coerce').dropna().tolist()

            if not prices or not volumes:
                continue

            avg_close = np.mean(prices)
            volatility = np.std(prices) # Standard deviation of prices as volatility
            avg_volume = np.mean(volumes)

            # Simulated risk category based on volatility for training the AI model
            if volatility < 10:
                risk = "Low"
            elif volatility < 50:
                risk = "Medium"
            else:
                risk = "High"

            data_rows.append([stock, avg_close, volatility, avg_volume, risk])
            time.sleep(0.5) # Be respectful to API limits

        except Exception as e:
            time.sleep(0.5) # Be respectful to API limits

    df = pd.DataFrame(data_rows, columns=["Stock", "Avg_Price", "Volatility", "Avg_Volume", "Risk"])
    return df

# ---------- STEP 2: TRAIN AI MODEL ----------

def train_model(df):
    """
    Trains a RandomForestClassifier model to predict stock risk categories
    based on average price, volatility, and average volume.
    """
    # Features for the model
    X = df[['Avg_Price', 'Volatility', 'Avg_Volume']]
    # Target variable (risk category)
    y = df['Risk']

    # Encode risk categories into numerical format for the model
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
    model.fit(X, y_encoded)

    return model, encoder

# ---------- STEP 3: FETCH LIVE STOCK PRICES ----------

def fetch_live_prices(stocks):
    """
    Fetches the latest live closing prices for a list of stock tickers.
    """
    prices = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            price_data = ticker.history(period="1d") # Get data for the last day
            # Extract the last available closing price
            price = price_data["Close"].dropna().iloc[-1] if not price_data.empty else None
            prices[stock] = round(price, 2) if price else None
            time.sleep(0.5) # Be respectful to API limits
        except Exception as e:
            prices[stock] = None
            time.sleep(0.5) # Be respectful to API limits
    return prices

# ---------- STEP 4: MAIN STOCK SUGGESTION TOOL ----------

def stock_suggestion_tool():
    """
    Main function for the stock suggestion tool.
    It fetches data, trains the model, gets live prices, and provides suggestions
    based on user's capital and minimum share requirements.
    """

    stock_list = [
        "TCS.NS", "RELIANCE.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "ICICIBANK.NS",
        "HDFCBANK.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
        "BHARTIARTL.NS", "MARUTI.NS", "HINDUNILVR.NS", "LT.NS", "AXISBANK.NS",
        "WIPRO.NS", "TECHM.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "TITAN.NS",
        "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "COALINDIA.NS",
        "GAIL.NS", "ADANIPORTS.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATAMOTORS.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "DMART.NS", "PIDILITIND.NS", "SRF.NS", "GODREJCP.NS", "DABUR.NS",
        "BRITANNIA.NS", "UPL.NS", "INDUSINDBK.NS", 
        "NIFTYBEES.NS", 
        "JUNIORBEES.NS", 
        "BANKBEES.NS",
        "CPSEETF.NS", 
        "MOMENTUM.NS", 
        "SETFGOLD.NS",
        "PETRONET.NS" 
    ]

    print("\n--- Stock Suggestion Tool ---")
    print(f"Please wait while the tool fetches data of stocks...")
    print("This process may take around 30-50 seconds depending on your internet speed...")
    print("Fetching historical stock data...")
    df = get_stock_data(stock_list)

    if df.empty:
        print("\n❌ Could not prepare dataset for stock suggestions. Returning to main menu.")
        return

    model, encoder = train_model(df)

    print("Fetching live stock prices...")
    live_prices = fetch_live_prices(stock_list)

    # Prepare data for risk prediction using the trained model
    prediction_data = df[['Avg_Price', 'Volatility', 'Avg_Volume']].values
    predicted_risk = model.predict(prediction_data)
    risk_categories = encoder.inverse_transform(predicted_risk)

    # Combine all stock information
    stock_info = []
    for i, stock in enumerate(stock_list):
        price = live_prices.get(stock)
        if price is None:
            # Fallback to average historical price if live price not available
            price_from_df = df.loc[df['Stock'] == stock, 'Avg_Price']
            price = round(price_from_df.values[0], 2) if not price_from_df.empty else None

        if price is None:
            continue

        stock_info.append({
            "Stock": stock,
            "Price": price,
            "Risk": risk_categories[i] if i < len(risk_categories) else "Unknown" # Handle potential mismatch
        })

    # --- User Input for Investment ---
    try:
        capital = float(input("\nEnter your total investment capital (₹): "))
        min_shares = int(input("Enter the minimum number of shares you want to buy (e.g., 1): "))
    except ValueError:
        print("Invalid input for capital or minimum shares. Please enter numeric values.")
        return

    # --- Filtering and Suggestions ---
    result = []
    for stock in stock_info:
        if stock["Price"] is None or stock["Price"] == 0:
            continue # Skip if price is invalid

        max_shares = int(capital // stock["Price"]) # Maximum shares affordable
        # Check if user can afford at least the minimum requested shares
        if max_shares >= min_shares:
            total_cost = round(max_shares * stock["Price"], 2)
            result.append([stock["Risk"], stock["Stock"], stock["Price"], max_shares, total_cost])

    # --- Output Suggestions ---
    if result:
        print("\n--- Stock Suggestions based on your criteria ---\n")
        # Sort by Risk (Low, Medium, High) and then by Total Cost
        risk_order = {"Low": 1, "Medium": 2, "High": 3}
        sorted_result = sorted(result, key=lambda x: (risk_order.get(x[0], 99), x[4]))
        print(tabulate(sorted_result, headers=["Risk", "Stock", "Price (₹)", "Max Shares", "Total Cost (₹)"], tablefmt="pretty"))
    else:
        print(f"\nNo stocks found matching your criteria (Capital: ₹{capital}, Min Shares: {min_shares}).")
        print("💡 Tip: Try lowering the minimum shares, increasing your capital, or running the tool again to see more options.")

# ---------- FUNCTIONS FROM NEWS.PY INTEGRATED BELOW ----------

def fetch_nse_stock_list():
    """
    Fetches a comprehensive list of NSE stocks from archives.nseindia.com.
    Used to build a mapping for fuzzy search.
    """
    stock_mapping = {}
    try:
        nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_nse = pd.read_csv(nse_url)
        for _, row in df_nse.iterrows():
            stock_mapping[row['NAME OF COMPANY'].lower()] = row['SYMBOL'] + ".NS"
            stock_mapping[row['SYMBOL'].lower()] = row['SYMBOL'] + ".NS" # Also map by symbol itself
    except Exception as e:
        print(f"Error fetching NSE stock list from archives.nseindia.com: {e}")
        print("Using a fallback list for stock lookup.")
        # Fallback to a hardcoded list if fetching fails
        fallback_stocks = {
            "reliance": "RELIANCE.NS", "tata motors": "TATAMOTORS.NS", "tcs": "TCS.NS",
            "infosys": "INFY.NS", "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS",
            "sbi": "SBIN.NS", "asian paints": "ASIANPAINT.NS", "kotak bank": "KOTAKBANK.NS",
            "bajaj finance": "BAJFINANCE.NS", "bharti airtel": "BHARTIARTL.NS", "maruti suzuki": "MARUTI.NS",
            "hindustan unilever": "HINDUNILVR.NS", "larsen & toubro": "LT.NS", "axis bank": "AXISBANK.NS",
            "wipro": "WIPRO.NS", "tech mahindra": "TECHM.NS", "nestle india": "NESTLEIND.NS",
            "ultratech cement": "ULTRACEMCO.NS", "titan": "TITAN.NS", "petronet": "PETRONET.NS",
            "gr infra": "GRINFRA.NS", "grinfra": "GRINFRA.NS"
        }
        stock_mapping.update(fallback_stocks)

    print(f"✅ Loaded {len(stock_mapping)} stocks for lookup.")
    return stock_mapping

def search_ticker_fuzzy(stock_name, stock_mapping):
    """
    Searches for a stock ticker using fuzzy matching against a predefined mapping.
    """
    if not stock_mapping:
        return None

    stock_name_lower = stock_name.lower()
    # Use fuzzywuzzy to find the best match
    result = process.extractOne(stock_name_lower, stock_mapping.keys())

    if result is None:
        return None

    best_match_key, score = result
    # A score of 80 or higher is generally considered a good match
    if score >= 80:
        return stock_mapping.get(best_match_key)
    else:
        return None

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text (e.g., news title).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def fetch_news(stock_query):
    """
    Fetches news articles for a given stock query using the NewsAPI.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY" or not NEWS_API_BASE_URL or NEWS_API_BASE_URL == "YOUR_NEWS_API_BASE_URL":
        return [[1, "News API not configured. Please set NEWS_API_KEY and NEWS_API_BASE_URL.", "", ""]]

    try:
        params = {
            'q': stock_query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'pageSize': 5, # Get top 5 articles
            'sortBy': 'publishedAt'
        }
        response = requests.get(NEWS_API_BASE_URL, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        news_data = []
        if articles:
            for idx, article in enumerate(articles, start=1):
                sentiment = analyze_sentiment(article.get("title", ""))
                title = article.get("title", "N/A").replace("\n", " ").strip()
                url_link = article.get("url", "")
                news_data.append([idx, title[:80] + ("..." if len(title) > 80 else ""), url_link, sentiment])
        else:
            news_data.append([1, "No recent news found.", "", ""])
        return news_data
    except requests.exceptions.RequestException as e:
        return [[1, f"Error fetching news: {e}. Check API key/URL and internet.", "", ""]]
    except Exception as e:
        return [[1, f"An unexpected error occurred while processing news: {e}", "", ""]]

def format_currency(value):
    """
    Formats a numerical value into a human-readable currency string (e.g., in Crores).
    """
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        value = float(value)
        return f"₹ {value / 1e7:.2f} Cr"
    except:
        return str(value)

def fetch_financials(stock_ticker):
    """
    Fetches essential financial data (revenue, PAT, market cap, PE, debt-to-equity, 52-week high/low).
    """
    stock = yf.Ticker(stock_ticker)

    try:
        # Fetch income statement (annual)
        income_stmt = stock.income_stmt.T
        
        # Get latest 3 years of revenue and net income
        revenue_data = {}
        pat_data = {}

        # Try common revenue keys
        revenue_keys = ['Total Revenue', 'Operating Revenue', 'Sales Revenue', 'Revenue']
        found_revenue_key = None
        for key in revenue_keys:
            if key in income_stmt.columns:
                found_revenue_key = key
                break

        if found_revenue_key:
            for i, (date, row) in enumerate(income_stmt.head(3).iterrows()): # Get latest 3 years
                year = date.year
                revenue_data[year] = format_currency(row.get(found_revenue_key))
        
        if 'Net Income' in income_stmt.columns:
            for i, (date, row) in enumerate(income_stmt.head(3).iterrows()): # Get latest 3 years
                year = date.year
                pat_data[year] = format_currency(row.get('Net Income'))

        info = stock.info
        essential_data = [
            ["Market Cap", format_currency(info.get("marketCap"))],
            ["PE Ratio", round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A"],
            ["Debt to Equity", round(info.get("debtToEquity"), 2) if info.get("debtToEquity") else "N/A"],
            ["52 Week High", format_currency(info.get("fiftyTwoWeekHigh"))],
            ["52 Week Low", format_currency(info.get("fiftyTwoWeekLow"))],
        ]

        financials_df_data = {}
        # Combine revenue and PAT data by year
        all_years = sorted(list(set(revenue_data.keys()) | set(pat_data.keys())), reverse=True)
        for year in all_years:
            financials_df_data[year] = {
                "Revenue": revenue_data.get(year, "N/A"),
                "PAT": pat_data.get(year, "N/A")
            }
        
        financials_df = pd.DataFrame.from_dict(financials_df_data, orient='index')
        financials_df.index.name = "Year"


        return {
            "Financial Table": financials_df,
            "Essential Metrics": essential_data
        }
    except Exception as e:
        print(f"Error fetching financials for {stock_ticker}: {e}")
        return {"error": f"Could not fetch financials: {e}"}

def analyze_vulnerability(stock_ticker):
    """
    Analyzes stock vulnerability based on historical volatility.
    """
    try:
        stock = yf.Ticker(stock_ticker)
        history = stock.history(period="1y") # 1 year historical data

        if history.empty or 'Close' not in history.columns:
            return [["Volatility (Annualized)", "N/A"], ["Predicted Risk Level", "N/A"], ["Vulnerability Analysis", "No historical data for analysis."]]

        # Calculate daily percentage change and then annualized volatility
        volatility = history['Close'].pct_change().std() * (252 ** 0.5) # 252 trading days in a year

        # Define risk levels based on volatility thresholds
        risk_level = "High" if volatility > 0.4 else "Medium" if volatility > 0.2 else "Low"

        return [
            ["Volatility (Annualized)", f"{round(volatility * 100, 2)}%"],
            ["Predicted Risk Level", risk_level]
        ]
    except Exception as e:
        print(f"Error analyzing vulnerability for {stock_ticker}: {e}")
        return [["Volatility (Annualized)", "Error"], ["Predicted Risk Level", "Error"], ["Vulnerability Analysis", f"Error: {e}"]]

def display_table(title, headers, rows):
    """
    Helper function to display data in a pretty table format.
    """
    print(f"\n===== {title} =====")
    table = PrettyTable()
    table.field_names = headers
    for row in rows:
        table.add_row(row)
    print(table)

def display_data(data):
    """
    Displays all fetched stock data (news, financials, vulnerability) in a structured format.
    """
    print("\n===== Stock Data =====")

    if "error" in data:
        print(data["error"])
        return

    display_table("News with Sentiment", ["S.No", "Title", "URL", "Sentiment"], data["News"])

    if "Financials" in data and isinstance(data["Financials"], dict):
        fin_data = data["Financials"]
        if "Financial Table" in fin_data and not fin_data["Financial Table"].empty:
            print("\n===== Financials (Annual) =====")
            print(fin_data["Financial Table"].to_markdown(index=True)) # Ensure index (Year) is printed
        else:
            print("\n===== Financials (Annual) =====")
            print("Financial table data not available.")

        if "Essential Metrics" in fin_data and fin_data["Essential Metrics"]:
            display_table("Essential Metrics", ["Metric", "Value"], fin_data["Essential Metrics"])
        else:
            print("\nEssential Metrics data not available.")
    else:
        print("\nFinancial data not available.")

    if "Vulnerability" in data and data["Vulnerability"]:
        display_table("Vulnerability Analysis", ["Metric", "Value"], data["Vulnerability"])
    else:
        print("\nVulnerability Analysis data not available.")


# ---------- MAIN STOCK DETAILS AND NEWS FUNCTION (REWRITTEN) ----------

def get_stock_details_and_news():
    """
    Allows the user to get detailed information and news about a specific stock.
    It uses fuzzy matching and direct yfinance lookup to resolve company names to tickers
    and provides comprehensive details using integrated news.py functions.
    """
    print("\n--- Get Stock Details and News ---")
    stock_input = input("Enter the stock name or ticker (e.g., 'Reliance', 'TCS.NS', 'Petronet'): ").strip()
    if not stock_input:
        print("No stock name entered. Returning to main menu.")
        return # Return to the main loop

    ticker_symbol = None

    # Try fuzzy matching first using the globally loaded stock_mapping
    print(f"Attempting to find ticker for '{stock_input}'...")
    ticker_symbol = search_ticker_fuzzy(stock_input, global_stock_mapping)

    if ticker_symbol:
        print(f"Fuzzy match found: {ticker_symbol}")
    else:
        # If fuzzy matching fails, try direct yfinance lookup as a fallback
        print(f"Fuzzy match failed. Trying direct yfinance lookup for '{stock_input}'...")
        try:
            temp_ticker = yf.Ticker(stock_input)
            temp_info = temp_ticker.info
            if temp_info and 'symbol' in temp_info:
                ticker_symbol = temp_info['symbol']
                print(f"Direct yfinance lookup found: {ticker_symbol}")
            else:
                print(f"Could not find a ticker for '{stock_input}' via direct lookup. Returning to main menu.")
                return # No ticker found, return to main menu
        except Exception as e:
            print(f"Error during direct yfinance lookup for '{stock_input}': {e}. Returning to main menu.")
            return # Error during direct lookup, return to main menu

    # If a ticker symbol is successfully identified, proceed to fetch and display data
    if ticker_symbol:
        print(f"Fetching detailed data for {ticker_symbol}...")       
        data = {
            "News": fetch_news(ticker_symbol),
            "Financials": fetch_financials(ticker_symbol),
            "Vulnerability": analyze_vulnerability(ticker_symbol)
        }
               
        display_data(data)
    else:
        print("No valid stock ticker could be identified. Returning to main menu.")

# ---------- RUN THE TOOL WITH POST-COMPLETION OPTIONS ----------

if __name__ == "__main__":
    # Load the comprehensive stock list for fuzzy matching once at startup
    print("Loading comprehensive stock list for news/details lookup...")
    global_stock_mapping = fetch_nse_stock_list()

    # Initial prompt for what the user wants to do
    while True:
        print("\n--- Welcome to the Stock Analysis Tool! ---")
        print("What would you like to do first?")
        print("1. Get Stock Suggestions (based on your capital/shares)")
        print("2. Get News and Details about a specific share")
        print("3. Exit")

        initial_choice = input("Enter your choice (1, 2, or 3): ").strip()

        if initial_choice == '1':
            stock_suggestion_tool()
            # After stock_suggestion_tool completes, the loop will continue to the main menu
        elif initial_choice == '2':
            get_stock_details_and_news()
            # After get_stock_details_and_news completes, the loop will continue to the main menu
        elif initial_choice == '3':
            print("Exiting the Stock Analysis Tool. Goodbye!")
            break # Exit the loop and end the program
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
