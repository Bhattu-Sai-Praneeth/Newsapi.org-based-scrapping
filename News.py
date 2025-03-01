import streamlit as st
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set your NewsAPI key
NEWSAPI_KEY = "063b1b2696c24c3a867c46c94cf9b810"

# Load FinBERT sentiment analysis model (caching for performance)
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

sentiment_pipeline = load_sentiment_model()

# Function to fetch the latest 10 news articles
def fetch_news(company_query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 10,  # Limit results to latest 10 articles
        "apiKey": NEWSAPI_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()
        return data.get("articles", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

# Streamlit app layout
st.title("News Sentiment Analysis with FinBERT")
st.write("Enter a company ticker or name to fetch the latest 10 news articles and analyze their sentiment.")

# Input field for company ticker/name
company_input = st.text_input("Company Ticker/Name", value="AAPL")

if st.button("Fetch News and Analyze Sentiment"):
    articles = fetch_news(company_input)
    
    if not articles:
        st.warning("No articles found.")
    else:
        st.success(f"Fetched {len(articles)} latest articles.")
        
        # Prepare a list to hold article info and sentiments
        results = []
        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""
            url = article.get("url") or ""
            
            # Combine title and description for sentiment analysis
            text_to_analyze = f"{title}. {description}"
            
            # Check if there's enough text to analyze
            if not text_to_analyze.strip() or text_to_analyze.strip() == ".":
                sentiment = "N/A"
            else:
                try:
                    result = sentiment_pipeline(text_to_analyze)[0]
                    sentiment = result["label"]  # Expected: 'positive', 'neutral', or 'negative'
                except Exception as e:
                    sentiment = "Error"
                    st.error(f"Sentiment analysis failed for: {title}. Error: {e}")
            
            results.append({
                "Title": title,
                "Description": description,
                "URL": url,
                "Sentiment": sentiment
            })
        
        # Create a DataFrame and display the results
        df = pd.DataFrame(results)
        st.write(df)
        
        # Display sentiment counts as a bar chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
