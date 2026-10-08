# --------------------------------------------------
# CUSTOMER REVIEW INTELLIGENCE DASHBOARD
#
# Run:
# python -m streamlit run app.py
# --------------------------------------------------

import html
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
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Global styling
# --------------------------------------------------

st.html(
    """
    <style>

    .stApp {
        background-color: #f7f8fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111827 0%,
                #1e1b4b 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background-color: white;

        border: 1px solid #e2e8f0;

        border-radius: 16px;

        padding: 1rem;

        box-shadow:
            0 4px 15px
            rgba(15, 23, 42, 0.05);
    }

    /* Buttons */

    div.stButton > button {
        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #6366f1
            );

        color: white;

        border: none;

        border-radius: 12px;

        font-weight: 700;

        padding: 0.65rem 1.2rem;
    }

    div.stButton > button:hover {
        color: white;

        box-shadow:
            0 8px 20px
            rgba(79, 70, 229, 0.25);
    }

    /* Dataframe */

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        overflow: hidden;
    }

    </style>
    """
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    reviews = pd.read_csv(
        "data/app_reviews.csv"
    )

    products = pd.read_csv(
        "data/products.csv"
    )

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
        model=(
            "SebasLopez-ai/"
            "distilbert-amazon-reviews-sentiment"
        )
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🛍️ ReviewIQ")

    st.caption(
        "Customer Review Intelligence"
    )

    st.write("")

    page = st.radio(
        "Navigation",
        [
            "📊 Category Insights",
            "🧭 Product Explorer",
            "💬 Review Analyzer"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption(
        "NLP Customer Review Project"
    )

    st.caption(
        "Sentiment • Clustering • Generative AI"
    )


# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.html(
    """
    <div style="
        background:
            linear-gradient(
                135deg,
                #ffffff 0%,
                #eef2ff 100%
            );

        border: 1px solid #e2e8f0;

        border-radius: 22px;

        padding: 38px;

        margin-bottom: 32px;

        box-shadow:
            0 10px 30px
            rgba(15,23,42,0.06);
    ">

        <div style="
            display:inline-block;

            background:#e0e7ff;
            color:#4338ca;

            border-radius:999px;

            padding:6px 12px;

            font-size:12px;

            font-weight:800;

            letter-spacing:1px;

            margin-bottom:14px;
        ">
            NLP CUSTOMER REVIEW PLATFORM
        </div>

        <div style="
            font-size:44px;

            line-height:1.1;

            font-weight:800;

            color:#0f172a;

            margin-bottom:12px;
        ">
            Customer Review Intelligence Dashboard
        </div>

        <div style="
            font-size:17px;

            color:#64748b;

            line-height:1.7;

            max-width:800px;
        ">
            Transform thousands of Amazon customer reviews
            into useful insights using sentiment analysis,
            product clustering and generative AI.
        </div>

    </div>
    """
)


# ==================================================
# PAGE 1
# CATEGORY INSIGHTS
# ==================================================

if page == "📊 Category Insights":

    st.caption(
        "OVERVIEW"
    )

    st.header(
        "Category Insights"
    )

    st.write(
        "Explore customer sentiment and AI-generated "
        "insights for each product category."
    )


    # --------------------------------------------------
    # Category selection
    # --------------------------------------------------

    category = st.selectbox(
        "Choose a product category",
        sorted(
            df[
                "meta_category"
            ]
            .dropna()
            .unique()
        )
    )


    category_df = df[
        df["meta_category"] == category
    ]


    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    total_reviews = len(
        category_df
    )

    unique_products = (
        category_df[
            "product_id"
        ]
        .nunique()
    )

    positive_pct = (
        category_df[
            "sentiment"
        ]
        .eq("Positive")
        .mean()
        * 100
    )

    neutral_pct = (
        category_df[
            "sentiment"
        ]
        .eq("Neutral")
        .mean()
        * 100
    )

    negative_pct = (
        category_df[
            "sentiment"
        ]
        .eq("Negative")
        .mean()
        * 100
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Products",
        f"{unique_products:,}"
    )

    col3.metric(
        "Positive",
        f"{positive_pct:.1f}%"
    )

    col4.metric(
        "Negative",
        f"{negative_pct:.1f}%"
    )


    st.write("")


    # --------------------------------------------------
    # Sentiment chart + AI summary
    # --------------------------------------------------

    chart_col, summary_col = st.columns(
        [1.4, 1],
        gap="large"
    )


    with chart_col:

        st.subheader(
            "Sentiment Distribution"
        )

        sentiment_percentage = (
            category_df[
                "sentiment"
            ]
            .value_counts(
                normalize=True
            )
            .reindex(
                [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                fill_value=0
            )
            * 100
        )

        sentiment_percentage = (
            sentiment_percentage
            .rename(
                "Percentage"
            )
        )

        st.bar_chart(
            sentiment_percentage,
            height=320
        )

        st.caption(
            f"Positive: {positive_pct:.1f}%"
            f"  •  "
            f"Neutral: {neutral_pct:.1f}%"
            f"  •  "
            f"Negative: {negative_pct:.1f}%"
        )


    with summary_col:

        summary = (
            category_summaries
            .get(
                category,
                "No summary available."
            )
        )

        safe_summary = html.escape(
            str(summary)
        )

        st.html(
            f"""
            <div style="
                background:
                    linear-gradient(
                        135deg,
                        #eef2ff,
                        #ffffff
                    );

                border:
                    1px solid #c7d2fe;

                border-radius:
                    18px;

                padding:
                    24px;

                min-height:
                    260px;

                box-shadow:
                    0 6px 20px
                    rgba(79,70,229,0.06);
            ">

                <div style="
                    color:#4f46e5;

                    font-size:12px;

                    font-weight:800;

                    letter-spacing:1px;

                    margin-bottom:10px;
                ">
                    ✨ GENERATIVE AI
                </div>

                <div style="
                    color:#312e81;

                    font-size:22px;

                    font-weight:800;

                    margin-bottom:14px;
                ">
                    AI Customer Summary
                </div>

                <div style="
                    color:#475569;

                    line-height:1.7;

                    font-size:15px;
                ">
                    {safe_summary}
                </div>

            </div>
            """
        )


    # --------------------------------------------------
    # Category information
    # --------------------------------------------------

    st.write("")

    safe_category = html.escape(
        str(category)
    )

    st.html(
        f"""
        <div style="
            background:white;

            border:
                1px solid #e2e8f0;

            border-radius:
                18px;

            padding:
                22px;

            box-shadow:
                0 5px 18px
                rgba(15,23,42,0.04);
        ">

            <div style="
                color:#6366f1;

                font-size:12px;

                font-weight:800;

                letter-spacing:1px;
            ">
                SELECTED CATEGORY
            </div>

            <div style="
                color:#0f172a;

                font-size:22px;

                font-weight:800;

                margin-top:6px;

                margin-bottom:8px;
            ">
                {safe_category}
            </div>

            <div style="
                color:#64748b;

                line-height:1.7;
            ">

                This category contains

                <strong>
                    {unique_products:,} products
                </strong>

                represented by

                <strong>
                    {total_reviews:,}
                    customer reviews
                </strong>.

            </div>

        </div>
        """
    )


# ==================================================
# PAGE 2
# PRODUCT EXPLORER
# ==================================================

elif page == "🧭 Product Explorer":

    st.caption(
        "PRODUCT CATALOG"
    )

    st.header(
        "Product Explorer"
    )

    st.write(
        "Browse products grouped into the "
        "meta-categories created by KMeans clustering."
    )


    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    filter_col, search_col = st.columns(
        2
    )


    with filter_col:

        category = st.selectbox(
            "Select category",
            sorted(
                products_df[
                    "meta_category"
                ]
                .dropna()
                .unique()
            )
        )


    with search_col:

        search_text = st.text_input(
            "Search product",
            placeholder=(
                "Search by product name or brand..."
            )
        )


    # --------------------------------------------------
    # Filter selected category
    # --------------------------------------------------

    category_products = products_df[
        products_df[
            "meta_category"
        ] == category
    ].copy()


    # --------------------------------------------------
    # Search filter
    # --------------------------------------------------

    if search_text.strip():

        product_match = (
            category_products[
                "product_name"
            ]
            .fillna("")
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        )

        brand_match = (
            category_products[
                "brand"
            ]
            .fillna("")
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        )

        category_products = (
            category_products[
                product_match
                |
                brand_match
            ]
        )


    # --------------------------------------------------
    # Clean missing names
    # --------------------------------------------------

    category_products[
        "product_name"
    ] = (
        category_products[
            "product_name"
        ]
        .fillna(
            "Name unavailable"
        )
        .replace(
            "Unknown",
            "Name unavailable"
        )
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    brand_count = (
        category_products[
            "brand"
        ]
        .dropna()
        .nunique()
    )


    metric1, metric2, metric3 = (
        st.columns(3)
    )


    metric1.metric(
        "Products Found",
        f"{len(category_products):,}"
    )


    metric2.metric(
        "Category",
        category
    )


    metric3.metric(
        "Brands",
        f"{brand_count:,}"
    )


    st.write("")


    # --------------------------------------------------
    # Table
    # --------------------------------------------------

    st.subheader(
        "Products in this Category"
    )


    display_df = (
        category_products[
            [
                "product_id",
                "product_name",
                "brand"
            ]
        ]
        .rename(
            columns={
                "product_id":
                    "Product ID",

                "product_name":
                    "Product Name",

                "brand":
                    "Brand"
            }
        )
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420
    )


    if len(
        category_products
    ) == 0:

        st.info(
            "No products match your search."
        )


# ==================================================
# PAGE 3
# REVIEW ANALYZER
# ==================================================

elif page == "💬 Review Analyzer":

    st.caption(
        "LIVE MACHINE LEARNING"
    )

    st.header(
        "Review Sentiment Analyzer"
    )

    st.write(
        "Enter a customer review and receive "
        "a live sentiment prediction from DistilBERT."
    )


    input_col, info_col = st.columns(
        [1.35, 0.65],
        gap="large"
    )


    # --------------------------------------------------
    # Input
    # --------------------------------------------------

    with input_col:

        review = st.text_area(
            "Enter a customer review",
            placeholder=(
                "Example: The tablet is easy to use, "
                "the screen looks great and "
                "the battery lasts all day."
            ),
            height=180
        )

        st.caption(
            f"{len(review.strip()):,} characters"
        )

        analyze_button = st.button(
            "Analyze Sentiment"
        )


    # --------------------------------------------------
    # Information card
    # --------------------------------------------------

    with info_col:

        st.html(
            """
            <div style="
                background:white;

                border:
                    1px solid #e2e8f0;

                border-radius:
                    18px;

                padding:
                    22px;

                box-shadow:
                    0 5px 18px
                    rgba(15,23,42,0.04);
            ">

                <div style="
                    color:#6366f1;

                    font-size:12px;

                    font-weight:800;

                    letter-spacing:1px;
                ">
                    HOW IT WORKS
                </div>

                <div style="
                    color:#0f172a;

                    font-size:20px;

                    font-weight:800;

                    margin-top:7px;

                    margin-bottom:12px;
                ">
                    Live DistilBERT Prediction
                </div>

                <div style="
                    color:#64748b;

                    line-height:1.7;
                ">

                    Your review is analyzed by a
                    pretrained Amazon DistilBERT model.

                    <br><br>

                    The model predicts:

                    <br><br>

                    • Positive<br>
                    • Neutral<br>
                    • Negative

                </div>

            </div>
            """
        )

        st.info(
            "The first prediction may take a little "
            "longer while the model is loaded."
        )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    if analyze_button:

        if review.strip():

            with st.spinner(
                "Analyzing customer sentiment..."
            ):

                sentiment_model = (
                    load_sentiment_model()
                )

                result = sentiment_model(
                    review,
                    truncation=True,
                    max_length=512
                )[0]


            sentiment = result[
                "label"
            ]

            confidence = float(
                result[
                    "score"
                ]
            )


            if sentiment == "Positive":

                sentiment_icon = "😊"

            elif sentiment == "Negative":

                sentiment_icon = "🙁"

            else:

                sentiment_icon = "😐"


            safe_sentiment = html.escape(
                str(sentiment)
            )


            st.html(
                f"""
                <div style="
                    background:
                        linear-gradient(
                            135deg,
                            #ffffff,
                            #eef2ff
                        );

                    border:
                        1px solid #c7d2fe;

                    border-radius:
                        18px;

                    padding:
                        24px;

                    margin-top:
                        20px;

                    box-shadow:
                        0 8px 24px
                        rgba(15,23,42,0.06);
                ">

                    <div style="
                        color:#6366f1;

                        font-size:12px;

                        font-weight:800;

                        letter-spacing:1px;
                    ">
                        PREDICTION
                    </div>

                    <div style="
                        color:#0f172a;

                        font-size:28px;

                        font-weight:800;

                        margin-top:8px;
                    ">

                        {sentiment_icon}
                        {safe_sentiment}

                    </div>

                    <div style="
                        color:#64748b;

                        margin-top:5px;
                    ">

                        Confidence:

                        <strong>
                            {confidence:.2%}
                        </strong>

                    </div>

                </div>
                """
            )


            st.progress(
                confidence
            )


        else:

            st.warning(
                "Please enter a review first."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.write("")

st.divider()

st.caption(
    "Customer Review Intelligence Dashboard "
    "· NLP Project · Version 1"
)