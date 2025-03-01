import streamlit as st
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set NewsAPI Key
NEWSAPI_KEY = "063b1b2696c24c3a867c46c94cf9b810"

# Load FinBERT sentiment model (cached for performance)
@st.cache_resource
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

sentiment_pipeline = load_finbert_model()

# Fetch latest 10 news articles for a company
def fetch_company_news(company_name):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 10,  # Get latest 10 articles
        "apiKey": NEWSAPI_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

# Streamlit UI Layout
st.set_page_config(page_title="Stock News Sentiment Analyzer", layout="wide")

st.title("📈 Company News Sentiment Analyzer")
st.write("Analyze the sentiment of the latest news articles for any company.")

# User input for company name
company_name = st.text_input("Enter Company Name (e.g., Apple Inc., Tesla, Microsoft)", value="Apple Inc.")

if st.button("Analyze News Sentiment"):
    articles = fetch_company_news(company_name)

    if not articles:
        st.warning("No recent news found for this company.")
    else:
        st.success(f"Fetched {len(articles)} latest news articles.")

        results = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            url = article.get("url", "")
            
            # Combine title + description for sentiment analysis
            text_to_analyze = f"{title}. {description}".strip()
            
            # Perform sentiment analysis if there is valid text
            sentiment = "N/A"
            if text_to_analyze and text_to_analyze != ".":
                try:
                    result = sentiment_pipeline(text_to_analyze)[0]
                    sentiment = result["label"]  # Expected: 'positive', 'neutral', 'negative'
                except Exception as e:
                    st.error(f"Sentiment analysis error: {e}")
                    sentiment = "Error"
            
            results.append({
                "Title": title,
                "Description": description,
                "URL": url,
                "Sentiment": sentiment
            })
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Display News Table
        st.subheader("📜 Latest News & Sentiment")
        st.dataframe(df.style.applymap(
            lambda x: "background-color: #d4edda" if x == "positive" else 
                      "background-color: #f8d7da" if x == "negative" else 
                      "background-color: #fff3cd" if x == "neutral" else "",
            subset=["Sentiment"]
        ))

        # Display Sentiment Distribution Chart
        sentiment_counts = df["Sentiment"].value_counts()
        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(sentiment_counts)
