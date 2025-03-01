import streamlit as st
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set your NewsAPI key (replace with your own key if needed)
NEWSAPI_KEY = "063b1b2696c24c3a867c46c94cf9b810"

# Function to load FinBERT sentiment analysis model (caching for performance)
@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

sentiment_pipeline = load_sentiment_model()

# Function to fetch news articles from NewsAPI.org for a given company query
def fetch_news(company_query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_query,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error("Error fetching news articles.")
        return []
    data = response.json()
    return data.get("articles", [])

# Streamlit app layout
st.title("News Sentiment Analysis with FinBERT")
st.write("Enter a company ticker or name to fetch recent news and analyze their sentiment.")

# Input field for company ticker/name
company_input = st.text_input("Company Ticker/Name", value="AAPL")

if st.button("Fetch News and Analyze Sentiment"):
    articles = fetch_news(company_input)
    
    if not articles:
        st.warning("No articles found.")
    else:
        st.success(f"Fetched {len(articles)} articles.")
        
        # Prepare a list to hold article info and sentiments
        results = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            url = article.get("url", "")
            
            # Combine title and description for sentiment analysis
            text_to_analyze = f"{title}. {description}"
            # Check if there is enough text to analyze
            if text_to_analyze.strip() == ".":
                sentiment = "N/A"
            else:
                result = sentiment_pipeline(text_to_analyze)[0]
                sentiment = result["label"]  # Expecting 'positive', 'neutral', or 'negative'
            
            results.append({
                "Title": title,
                "Description": description,
                "URL": url,
                "Sentiment": sentiment
            })
        
        # Create a DataFrame and display the results
        df = pd.DataFrame(results)
        st.write(df)
        
        # Optional: Display sentiment counts as a bar chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
