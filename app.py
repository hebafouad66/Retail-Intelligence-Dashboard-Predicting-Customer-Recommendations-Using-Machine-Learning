import csv
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "recommended_product_model.pkl"
DATA_PATH = BASE_DIR / "Womens-E-Commerce-Clothing-Reviews (2).arff"

st.set_page_config(
    page_title="Retail Recommendation Dashboard",
    page_icon="⭐",
    layout="wide",
)

st.markdown(
    """
    <style>
    body {
        background: #f3f7fb;
        color: #0f172a;
    }
    .stApp > header {
        background: linear-gradient(90deg, #4933ff, #1e3a8a);
    }
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        color: white;
        font-weight: 700;
        padding: 0.8rem 1.2rem;
        border: none;
    }
    .result-card {
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        margin-bottom: 18px;
    }
    .success-card {
        background: radial-gradient(circle at top left, #d1fae5, #10b981 35%, #047857 100%);
        color: #063528;
    }
    .error-card {
        background: radial-gradient(circle at top left, #fee2e2, #ef4444 35%, #991b1b 100%);
        color: #7f1d1d;
    }
    .metric-box {
        background: white;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .footer-text {
        color: #64748b;
        font-size: 0.95rem;
        text-align: center;
        margin-top: 24px;
    }
    .highlight-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }
    .contrast-box {
        background: #eef2ff;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid #c7d2fe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(
    "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=1650&q=80",
    width=1100,
)
st.title("Retail Recommendation & Customer Satisfaction Dashboard")
st.markdown(
    "A polished multi-page dashboard built for recommendation prediction, satisfaction analytics, and NLP review insights."
)
page = st.sidebar.radio(
    "Choose a dashboard section",
    [
        "🏠 Executive Dashboard",
        "⭐ Predict Recommendation",
        "📊 EDA Dashboard",
        "🛠 Feature Engineering",
        "📈 Model Performance",
        "💬 NLP Analysis",
        "📋 Business Recommendations"
    ],
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found. Place 'recommended_product_model.pkl' in {BASE_DIR}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reviews():
    if not DATA_PATH.exists():
        return None

    attributes = []
    rows = []
    in_data = False
    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as arff_file:
        for raw_line in arff_file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) >= 2:
                    attributes.append(parts[1])
            elif line.lower().startswith("@data"):
                in_data = True
                continue
            elif in_data:
                reader = csv.reader(
                    [line], quotechar="'", escapechar="\\", skipinitialspace=True
                )
                item = next(reader)
                rows.append([None if value == "?" else value for value in item])

    df = pd.DataFrame(rows, columns=attributes)
    numeric_columns = [
        "Age",
        "Rating",
        "Recommended_IND",
        "Positive_Feedback_Count",
        "Clothing_ID",
        "Unnamed:_0",
    ]
    for name in numeric_columns:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")

    if "Title" in df.columns:
        df["Title"] = df["Title"].fillna("No Title")
    if "Review_Text" in df.columns:
        df["Review_Text"] = df["Review_Text"].fillna("No Review")

    df["Is_Young"] = (df["Age"] < 30).astype(int)
    df["Opinion_Strength"] = (df["Rating"] - 3).abs()
    df["Review_Word_Count"] = df["Review_Text"].astype(str).str.split().str.len()
    feedback_median = df["Positive_Feedback_Count"].median()
    df["Engagement"] = df["Positive_Feedback_Count"].apply(
        lambda value: "High" if value > feedback_median else "Low"
    )
    df["Customer_Type"] = df["Rating"].apply(
        lambda rating: "Happy"
        if rating >= 4
        else ("Unhappy" if rating <= 2 else "Neutral")
    )
    df["Risk"] = ((df["Rating"] <= 2) & (df["Recommended_IND"] == 0)).astype(int)
    return df


def build_prediction_input(
    age,
    positive_feedback_count,
    rating,
    division_name,
    department_name,
    class_name,
    feedback_median,
):
    review_word_count = 0
    is_young = int(age < 30)
    opinion_strength = abs(int(rating) - 3)
    customer_type = (
        "Happy"
        if rating >= 4
        else "Unhappy"
        if rating <= 2
        else "Neutral"
    )
    engagement = "High" if positive_feedback_count > feedback_median else "Low"
    risk = int(rating <= 2)

    return pd.DataFrame(
        {
            "Age": [age],
            "Positive_Feedback_Count": [positive_feedback_count],
            "Rating": [rating],
            "Division_Name": [division_name],
            "Department_Name": [department_name],
            "Class_Name": [class_name],
            "Review_Word_Count": [review_word_count],
            "Opinion_Strength": [opinion_strength],
            "Is_Young": [is_young],
            "Customer_Type": [customer_type],
            "Engagement": [engagement],
            "Risk": [risk],
            "Clothing_ID": [0],
        }
    )


def normalize_text(text):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    token_list = cleaned.split()
    stopwords = {
        "the",
        "and",
        "this",
        "that",
        "with",
        "was",
        "for",
        "have",
        "they",
        "but",
        "not",
        "from",
        "just",
        "very",
        "item",
        "it",
        "because",
        "would",
        "really",
        "there",
        "still",
        "them",
        "been",
        "when",
        "were",
        "than",
        "also",
        "can",
        "got",
        "one",
        "so",
        "if",
    }
    return [token for token in token_list if token.isalpha() and token not in stopwords]


def top_review_words(df, rating_condition):
    filtered = df.loc[rating_condition, "Review_Text"].dropna().astype(str)
    tokens = Counter()
    for review in filtered:
        tokens.update(normalize_text(review))
    return tokens.most_common(14)


reviews_df = load_reviews()
model = None
try:
    model = load_model()
except Exception as error:
    st.error(str(error))

if reviews_df is not None:
    division_options = sorted(reviews_df["Division_Name"].dropna().unique())
    department_options = sorted(reviews_df["Department_Name"].dropna().unique())
    class_options = sorted(reviews_df["Class_Name"].dropna().unique())
    median_feedback = int(reviews_df["Positive_Feedback_Count"].median())
else:
    division_options = ["General", "General Petite", "Initmates"]
    department_options = ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"]
    class_options = [
        "Dresses",
        "Knits",
        "Blouses",
        "Sweaters",
        "Pants",
        "Jeans",
        "Fine gauge",
        "Skirts",
        "Jackets",
        "Lounge",
        "Swim",
        "Shorts",
        "Legwear",
        "Intimates",
        "Sleep",
        "Outerwear",
    ]
    median_feedback = 4
if page == "🏠 Executive Dashboard":

    st.header("Executive Dashboard")

    if reviews_df is None:
        st.warning("Review dataset not available. Executive Dashboard requires the ARFF dataset.")
    else:
        st.info("""
        Business Problem

        Predict whether a customer will recommend a product
        based on customer demographics, review behavior,
        engagement level and satisfaction indicators.

        Business Value

        • Improve customer satisfaction
        • Detect risky customers
        • Increase recommendation rate
        • Understand customer behavior
        """)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Reviews",
            f"{len(reviews_df):,}",
        )

        col2.metric(
            "Recommendation Rate",
            f"{reviews_df['Recommended_IND'].mean()*100:.1f}%",
        )

        col3.metric(
            "Average Rating",
            round(reviews_df['Rating'].mean(), 2),
        )

        col4.metric(
            "Average Review Length",
            round(reviews_df['Review_Word_Count'].mean(), 1),
        )

        st.markdown("---")

        fig = px.pie(
            reviews_df,
            names="Recommended_IND",
            title="Target Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Project Objective

        Build a machine learning model capable of predicting
        customer recommendation behavior using:

        - Customer demographics
        - Product category
        - Review characteristics
        - Customer engagement
        - Satisfaction indicators
        """)

if page == "⭐ Predict Recommendation":
    st.header("Predict Recommendation")
    st.write(
        "Enter the exact customer and product attributes used to train the recommendation pipeline. "
        "Derived engineered features are computed automatically to keep the pipeline consistent."
    )

    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        age = st.slider("Age", min_value=18, max_value=90, value=33)
        positive_feedback = st.number_input(
            "Positive Feedback Count",
            min_value=0,
            max_value=500,
            value=5,
            step=1,
        )
    with input_col2:
        rating = st.select_slider(
            "Rating",
            options=[1, 2, 3, 4, 5],
            value=4,
        )
        division_name = st.selectbox("Division Name", division_options)
    with input_col3:
        department_name = st.selectbox("Department Name", department_options)
        class_name = st.selectbox("Class Name", class_options)

    st.write("---")
    if st.button("Run Recommendation"):
        if model is None:
            st.error("Model failed to load. Please make sure recommended_product_model.pkl exists.")
        else:
            request_df = build_prediction_input(
                age,
                positive_feedback,
                rating,
                division_name,
                department_name,
                class_name,
                median_feedback,
            )
            prediction = model.predict(request_df)[0]
            proba = model.predict_proba(request_df)[0][1]

            if prediction == 1:
                st.markdown(
                    """
                    <div class='result-card success-card'>
                        <h2>✅ Recommendation Predicted</h2>
                        <p>Customer signals and product context point to a likely recommendation.</p>
                        <p><strong>Confidence:</strong> {:.2f}%</p>
                    </div>
                    """.format(proba * 100),
                    unsafe_allow_html=True,
                )
                st.balloons()
            else:
                st.markdown(
                    """
                    <div class='result-card error-card'>
                        <h2>⚠️ Recommendation Not Recommended</h2>
                        <p>The pipeline predicts this profile is less likely to recommend the product.</p>
                        <p><strong>Confidence:</strong> {:.2f}%</p>
                    </div>
                    """.format((1 - proba) * 100),
                    unsafe_allow_html=True,
                )

            st.write("### Model Input Summary")
            st.dataframe(request_df.T, use_container_width=True)
elif page == "📊 EDA Dashboard":
    st.header("Exploratory Data Analysis")

    if reviews_df is None:
        st.warning("Review dataset not available. EDA visualizations require the local ARFF dataset.")
    else:
        missing = (
            reviews_df
            .isnull()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        missing.columns = ["Feature", "Missing %"]

        fig_missing = px.bar(
            missing,
            x="Feature",
            y="Missing %",
            title="Missing Values Analysis",
            color="Missing %",
        )
        st.plotly_chart(fig_missing, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(
                reviews_df,
                x="Age",
                title="Age Distribution",
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            fig_rating = px.histogram(
                reviews_df,
                x="Rating",
                color="Recommended_IND",
                title="Rating Distribution",
            )
            st.plotly_chart(fig_rating, use_container_width=True)

        fig_feedback = px.histogram(
            reviews_df,
            x="Positive_Feedback_Count",
            title="Feedback Distribution",
        )
        st.plotly_chart(fig_feedback, use_container_width=True)


elif page == "🛠 Feature Engineering":
    st.header("Feature Engineering")

    if reviews_df is None:
        st.warning("Review dataset not available. Feature engineering previews require the dataset.")
    else:
        engineered = [
            "Opinion_Strength",
            "Customer_Type",
            "Engagement",
            "Risk",
            "Is_Young",
            "Review_Word_Count",
        ]

        st.dataframe(
            reviews_df[engineered].head(20),
            use_container_width=True,
        )

        fig_words = px.histogram(
            reviews_df,
            x="Review_Word_Count",
            color="Recommended_IND",
            title="Review Length Impact",
        )
        st.plotly_chart(fig_words, use_container_width=True)

        fig_type = px.bar(
            reviews_df["Customer_Type"].value_counts().reset_index(name="count"),
            x="Customer_Type",
            y="count",
            title="Customer Types",
            color="count",
        )
        st.plotly_chart(fig_type, use_container_width=True)

elif page == "📈 Model Performance":
    st.header("Model Performance")

    models_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "KNN",
                "Decision Tree",
                "Random Forest",
                "SVM",
            ],
            "F1 Score": [
                96.33,
                95.16,
                95.29,
                96.00,
                90.24,
            ],
        }
    )

    fig = px.bar(
        models_df,
        x="Model",
        y="F1 Score",
        color="F1 Score",
        text="F1 Score",
        title="Model Comparison by F1 Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        """
    Best Model: Logistic Regression

    F1 Score = 96.33%

    Strong generalization
    Minimal overfitting
    Fast inference
    Suitable for deployment
    """
    )

elif page == "💬 NLP Analysis":
    st.header("NLP Analysis")
    st.write(
        "This section recreates the notebook's full text-cleaning and review-linguistics analysis with all the exact steps and visuals."
    )

    if reviews_df is None:
        st.warning("Review dataset not available. Create a local ARFF file to enable NLP insights.")
    else:
        title_missing = (reviews_df["Title"] == "No Title").sum()
        review_missing = (reviews_df["Review_Text"] == "No Review").sum()

        st.markdown("### Text Cleaning Process")
        st.markdown(
            "- `Title` and `Review_Text` are both filled with defaults using `fillna('No Title')` and `fillna('No Review')` to avoid missing values."
            "\n- The review text is normalized to lowercase and punctuation is removed."
            "\n- Stopwords and common filler tokens are excluded before counting the most frequent terms."
            "\n- `Review_Word_Count` is computed from the cleaned text and used as a review intensity feature in the model."
            "\n- The notebook analysis shows that missing text values are handled first, then text is tokenized and counted for sentiment signal."
        )

        st.markdown(
            f"- Missing title records filled: **{title_missing}**."
            f"\n- Missing review text records filled: **{review_missing}**."
        )

        st.markdown("### Review Length Distribution")
        fig_review_length = px.histogram(
            reviews_df,
            x="Review_Word_Count",
            nbins=40,
            title="Review Word Count Distribution",
            color_discrete_sequence=["#8b5cf6"],
        )
        st.plotly_chart(fig_review_length, use_container_width=True)

       

        st.markdown("### Linguistic Insights")
        st.markdown(
            "- Positive reviews often contain terms like **love**, **great**, **comfortable**, **perfect**, and **cute**."
            "\n- Negative reviews frequently include words such as **small**, **cheap**, **poor**, **tight**, and **return**."
            "\n- The review corpus uses strong sentiment terms, meaning `Opinion_Strength` captures more than just a numerical rating; it signals how clearly a customer feels."
            "\n- The notebook confirms that review-term frequency and review length are both meaningful behavioral signals."
        )

        st.markdown("### Review Examples from the Dataset")
        sample_rows = reviews_df.sample(3, random_state=42)[["Title", "Review_Text", "Rating"]]
        for _, row in sample_rows.iterrows():
            st.markdown(
                f"**Rating {int(row['Rating'])}** — *{row['Title']}*\n\n{row['Review_Text'][:320]}..."
            )

elif page == "📋 Business Recommendations":
    st.header("Business Recommendations")

    st.success(
        """
    Key Findings

    1. Rating is the strongest predictor.

    2. Customers with ratings <= 2 are
    considered high risk.

    3. Longer reviews contain richer
    behavioral signals.

    4. Dresses generate the largest
    review volume.

    5. Positive feedback count is strongly
    associated with recommendation behavior.
    """
    )

    st.warning(
        """
    Recommended Actions

    • Monitor low-rating products.
    • Prioritize customer complaints.
    • Improve underperforming categories.
    • Encourage detailed reviews.
    • Focus marketing on high satisfaction segments.
    """
    )

st.markdown("---")
st.markdown("<div class='footer-text'>Developed by Heba</div>", unsafe_allow_html=True)
