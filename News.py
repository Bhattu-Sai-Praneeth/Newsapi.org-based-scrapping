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
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            url = article.get("url", "#")
            
            # Perform sentiment analysis
            text_to_analyze = f"{title}. {description}".strip()
            sentiment = "N/A"
            if text_to_analyze and text_to_analyze != ".":
                try:
                    result = sentiment_pipeline(text_to_analyze)[0]
                    sentiment = result["label"]  # Expected: 'positive', 'neutral', 'negative'
                except Exception as e:
                    st.error(f"Sentiment analysis error: {e}")
                    sentiment = "Error"
            
            # Store results
            results.append({"Title": title, "URL": url, "Description": description, "Sentiment": sentiment})
        
        # Display Results with Clickable Links
        st.subheader("📜 Latest News & Sentiment")
        for news in results:
            st.markdown(f"### [{news['Title']}]({news['URL']})")
            st.write(f"**Sentiment:** {news['Sentiment']}")
            st.write(f"*{news['Description']}*")
            st.write("---")

        # Display Sentiment Distribution Chart
        sentiment_counts = pd.DataFrame([news["Sentiment"] for news in results], columns=["Sentiment"]).value_counts()
        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(sentiment_counts)
