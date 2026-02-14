# ======================== IMPORTS ========================
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import pandas as pd
import altair as alt
import os

# ✅ ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np


# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="AI Career Guidance System | SDG 8",
    page_icon="🎯",
    layout="wide"
)

# ======================== LOAD MODEL ========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ======================== LOAD DATASET ========================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("career_skills_dataset_levelwise_skills.csv", encoding="utf-8")
        if df.empty:
            st.error("⚠ Dataset is empty. Please check the file.")
        return df
    except Exception as e:
        st.error("❌ Dataset could not be loaded.")
        st.stop()

df = load_dataset()

# ======================== ML MODEL TRAINING (IMPROVED) ========================
@st.cache_data
def train_ml_model(df):
    # 1. Text clean
    df["skills"] = df["skills"].astype(str).str.lower()

    # 2. Label Encode career
    le = LabelEncoder()
    df["career_label"] = le.fit_transform(df["career"])

    # 3. Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["skills"], df["career_label"], test_size=0.2, random_state=42
    )

    # 4. Embeddings
    X_train_emb = np.array([model.encode(x) for x in X_train])
    X_test_emb = np.array([model.encode(x) for x in X_test])

    # 5. XGBoost Model
    model_ml = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_ml.fit(X_train_emb, y_train)

    # 6. Accuracy
    y_pred = model_ml.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)

    return model_ml, le, acc

model_ml, le, acc = train_ml_model(df)


# ======================== SIDEBAR ========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📄 Resume Analyzer", "🧠 Career Match", "📊 Skill Roadmap"]
)

# ============================================================
# 🏠 HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding:30px;">
        <h1>🎯 AI-Powered Career Guidance System</h1>
        <h3 style="color:gray;">Supporting UN SDG 8 – Decent Work & Economic Growth</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.info("📄 **AI Resume Analyzer**\n\nExplainable ATS-style scoring")
    col2.info("🧠 **Career Recommendation**\n\nDataset + NLP based matching")
    col3.info("🚀 **Skill Roadmap**\n\nDataset-driven personalized roadmap")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.success("🌍 End-to-end Applied AI Project (Internship Ready)")
    st.warning("⚠ This tool provides guidance only and does not guarantee job placement. No personal data is stored.")


# ============================================================
# 📄 RESUME ANALYZER
# ============================================================
elif page == "📄 Resume Analyzer":
    st.header("📄 AI Resume Analyzer (Explainable AI)")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file:
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""

        for p in reader.pages:
            if p.extract_text():
                resume_text += p.extract_text().lower()

        if not resume_text.strip():
            st.error("❌ Unable to extract text from the uploaded resume.")
            st.stop()

        st.subheader("📌 Resume Preview")
        st.text_area("", resume_text[:2500], height=220)

        technical = (
            df["skills"]
            .str.lower()
            .str.split(",")
            .explode()
            .dropna()
            .unique()
            .tolist()
        )

        soft_skills = ["communication", "teamwork", "leadership", "problem solving"]
        experience_words = ["project", "internship", "experience"]

        def score_section(words, weight):
            scores = []
            for w in words:
                e1 = model.encode(resume_text, convert_to_tensor=True)
                e2 = model.encode(w, convert_to_tensor=True)
                scores.append(util.cos_sim(e1, e2).item())
            return round((sum(scores) / len(scores)) * weight)

        tech_score = score_section(technical[:20], 40)
        soft_score = score_section(soft_skills, 20)
        exp_score = score_section(experience_words, 20)
        edu_score = 20 if any(x in resume_text for x in ["degree", "b.tech", "bachelor"]) else 8

        final_score = min(100, tech_score + soft_score + exp_score + edu_score)

        st.subheader("📊 ATS Resume Score")
        st.progress(final_score / 100)

        st.write(f"""
        **Final Score:** {final_score}/100  
        - Technical Skills: {tech_score}/40  
        - Soft Skills: {soft_score}/20  
        - Experience: {exp_score}/20  
        - Education: {edu_score}/20  
        """)

        st.info("ℹ Resume scoring is based on semantic similarity using NLP embeddings.")
        st.success("💡 Tip: Add role-specific projects & measurable achievements")

# ============================================================
# 🧠 CAREER MATCH (IMPROVED ML MODEL)
# ============================================================
elif page == "🧠 Career Match":
    st.header("🧠 AI Career Recommendation System (ML Model)")

    user_skills = st.text_area(
        "Enter your skills (comma separated)",
        placeholder="python, sql, machine learning"
    ).lower()

    if st.button("🔍 Find Best Career") and user_skills:
        # Convert user skills to embedding
        user_emb = model.encode(user_skills).reshape(1, -1)

        # Predict probability
        probs = model_ml.predict_proba(user_emb)[0]

        # Top 3 predictions
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]

        st.subheader("🏆 Top Career Predictions")
        for i, (career, prob) in enumerate(top3, 1):
            st.success(f"#{i} {career} — {round(prob*100, 2)}%")

        chart_df = pd.DataFrame({
            "Career": [x[0] for x in top3],
            "Probability": [round(x[1]*100, 2) for x in top3]
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x="Career",
            y="Probability"
        )

        st.altair_chart(chart, use_container_width=True)

# ============================================================
# 📊 SKILL ROADMAP
# ============================================================
elif page == "📊 Skill Roadmap":
    st.header("📊 Personalized Skill Roadmap")

    career = st.selectbox(
        "Select Career",
        sorted(df["career"].unique())
    )

    career_df = df[df["career"] == career]

    st.subheader(f"📌 Roadmap for {career}")

    for level, months in [
        ("Beginner", "0–2 months"),
        ("Intermediate", "3–5 months"),
        ("Advanced", "6–9 months")
    ]:
        st.markdown(f"### {level} ({months})")

        skills = career_df[
            career_df["experience_level"] == level
        ]["skills"].tolist()

        if skills:
            for s in skills:
                st.success(s)
        else:
            st.info("No data available for this level.")
