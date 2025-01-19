import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import re
from gensim import corpora, models
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
import spacy
from spacy import displacy

def fetch_news_data(ticker1, ticker2):
    """
    Fetches news data for the given tickers from Yahoo Finance.
    """
    try:
        ticker1_data = yf.Ticker(ticker1).news
        ticker2_data = yf.Ticker(ticker2).news
        return ticker1_data, ticker2_data
    except Exception as e:
        st.error(f"An error occurred while fetching news data: {e}")
        return fetch_news_data()

def clean_news_data(news_data):
    """
    Cleans the news data by removing special characters and extra spaces.
    """
    clean_data = []
    for news in news_data:
        news_text = re.sub(r"[^\w\s]", "", news['title'])
        news_text = " ".join(news_text.split())
        clean_data.append(news_text)
    return clean_data

def perform_sentiment_analysis(news_data):
    """
    Performs sentiment analysis on the given news data using TextBlob.
    """
    sentiments = []
    for news in news_data:
        analysis = TextBlob(news)
        sentiments.append(analysis.sentiment.polarity)
    return sentiments

def classify_sentiment(sentiment_scores):
    """
    Classifies sentiment based on the polarity scores.
    """
    sentiments = []
    for score in sentiment_scores:
        if score > 0:
            sentiments.append('Positive')
        elif score < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    return sentiments

def topic_modeling(news_data):
    """
    Performs topic modeling using LDA.
    """
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokenized_news = [word_tokenize(text) for text in news_data]
    tokenized_news = [[word for word in tokens if not word in stop_words] for tokens in tokenized_news]

    # Create Dictionary and Corpus
    dictionary = corpora.Dictionary(tokenized_news)
    corpus = [dictionary.doc2bow(text) for text in tokenized_news]

    # Train LDA model
    lda_model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

    return lda_model, dictionary

def extract_topics(lda_model, dictionary, num_words=5):
    """
    Extracts top topics from the LDA model.
    """
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topic_words = " ".join([word for word, _ in dictionary.items() if word in topic])
        topics.append(f"Topic {idx}: {topic_words}")
    return topics

def topic_sentiment_analysis(news_data, lda_model, dictionary):
    """
    Performs sentiment analysis for each topic.
    """
    topic_sentiments = {}
    for news, topic_idx in zip(news_data, lda_model.get_document_topics()):
        topic_idx = max(topic_idx, key=lambda x: x[1])[0]
        if topic_idx not in topic_sentiments:
            topic_sentiments[topic_idx] = []
        topic_sentiments[topic_idx].append(TextBlob(news).sentiment.polarity)

    return topic_sentiments

def ner_and_event_detection(news_data):
    """
    Performs Named Entity Recognition and Event Detection.
    """
    nlp = spacy.load("en_core_web_sm")
    events = []
    for news in news_data:
        doc = nlp(news)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        events.extend(entities)

    return events

def display_results(ticker1, ticker2, ticker1_sentiments, ticker2_sentiments,
                    ticker1_topics, ticker2_topics, 
                    ticker1_topic_sentiments=None, ticker2_topic_sentiments=None):
    """
    Displays the sentiment analysis results in Streamlit.

    Args:
        ticker1: The first ticker symbol.
        ticker2: The second ticker symbol.
        ticker1_sentiments: A list of sentiment classifications for ticker1.
        ticker2_sentiments: A list of sentiment classifications for ticker2.
        ticker1_topics: A list of topics for ticker1.
        ticker2_topics: A list of topics for ticker2.
        ticker1_topic_sentiments: A dictionary of topic sentiments for ticker1.
        ticker2_topic_sentiments: A dictionary of topic sentiments for ticker2.
    """
    st.title(f"{ticker1} vs. {ticker2} Sentiment Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.header(f"{ticker1} Sentiment")
        st.bar_chart(pd.Series(ticker1_sentiments).value_counts())
        st.header(f"{ticker1} Topics")
        for topic in ticker1_topics:
            st.write(topic)
        if ticker1_topic_sentiments:
            st.header(f"{ticker1} Topic Sentiments")
            for topic_idx, sentiments in ticker1_topic_sentiments.items():
                st.write(f"Topic {topic_idx}: Average Sentiment = {sum(sentiments) / len(sentiments)}")

    with col2:
        st.header(f"{ticker2} Sentiment")
        st.bar_chart(pd.Series(ticker2_sentiments).value_counts())
        st.header(f"{ticker2} Topics")
        for topic in ticker2_topics:
            st.write(topic)
        if ticker2_topic_sentiments:
            st.header(f"{ticker2} Topic Sentiments")
            for topic_idx, sentiments in ticker2_topic_sentiments.items():
                st.write(f"Topic {topic_idx}: Average Sentiment = {sum(sentiments) / len(sentiments)}")

    return display_results