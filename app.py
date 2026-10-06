
#to run the app install : pip install streamlit
# then this command : streamlit run app.py

import json
import pandas as pd
import streamlit as st
from transformers import pipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="🛍️",
    layout="wide"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    reviews = pd.read_csv("data/app_reviews.csv")
    products = pd.read_csv("data/products.csv")

    with open(
        "category_summaries.json",
        "r",
        encoding="utf-8"
    ) as f:
        summaries = json.load(f)

    return reviews, products, summaries


df, products_df, category_summaries = load_data()


# --------------------------------------------------
# Load sentiment model
# --------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="SebasLopez-ai/distilbert-amazon-reviews-sentiment"
    )


sentiment_model = load_sentiment_model()


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("Customer Review Intelligence Dashboard")

st.write(
    "Explore Amazon product categories, customer sentiment, "
    "AI-generated review summaries, and analyze individual reviews."
)


# --------------------------------------------------
# Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Category Insights",
        "Product Explorer",
        "Review Analyzer"
    ]
)


# --------------------------------------------------
# PAGE 1 - CATEGORY INSIGHTS
# --------------------------------------------------

if page == "Category Insights":

    st.header("Category Insights")

    category = st.selectbox(
        "Choose a product category",
        sorted(df["meta_category"].dropna().unique())
    )

    category_df = df[
        df["meta_category"] == category
    ]

    total_reviews = len(category_df)

    positive_pct = (
        category_df["sentiment"]
        .eq("Positive")
        .mean() * 100
    )

    neutral_pct = (
        category_df["sentiment"]
        .eq("Neutral")
        .mean() * 100
    )

    negative_pct = (
        category_df["sentiment"]
        .eq("Negative")
        .mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Positive",
        f"{positive_pct:.1f}%"
    )

    col3.metric(
        "Neutral",
        f"{neutral_pct:.1f}%"
    )

    col4.metric(
        "Negative",
        f"{negative_pct:.1f}%"
    )

    st.subheader("Sentiment Distribution")

    sentiment_counts = (
        category_df["sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
    )

    st.bar_chart(sentiment_counts)

    st.subheader("AI Customer Summary")

    summary = category_summaries.get(
        category,
        "No summary available."
    )

    st.info(summary)


# --------------------------------------------------
# PAGE 2 - PRODUCT EXPLORER
# --------------------------------------------------

elif page == "Product Explorer":

    st.header("Product Explorer")

    category = st.selectbox(
        "Select category",
        sorted(
            products_df["meta_category"]
            .dropna()
            .unique()
        )
    )

    category_products = products_df[
        products_df["meta_category"] == category
    ].copy()

    st.write(
        f"Products found: {len(category_products)}"
    )

    display_columns = [
        "product_id",
        "product_name",
        "brand"
    ]

    st.dataframe(
        category_products[display_columns],
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PAGE 3 - REVIEW ANALYZER
# --------------------------------------------------

elif page == "Review Analyzer":

    st.header("Review Sentiment Analyzer")

    review = st.text_area(
        "Enter a customer review",
        placeholder=(
            "Example: The tablet is easy to use "
            "and the battery lasts a long time."
        )
    )

    if st.button("Analyze Sentiment"):

        if review.strip():

            result = sentiment_model(review)[0]

            sentiment = result["label"]
            confidence = result["score"]

            st.subheader("Prediction")

            st.write(
                f"**Sentiment:** {sentiment}"
            )

            st.write(
                f"**Confidence:** {confidence:.2%}"
            )

        else:
            st.warning(
                "Please enter a review first."
            )