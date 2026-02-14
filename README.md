# 🎯 AI-Powered Career Guidance System

An end-to-end AI application that provides career recommendations, resume analysis, and personalized skill roadmaps using NLP and Machine Learning.

🔴 **Live App:**  
https://ai-career-guidance-system-drmkhttknffsjwwbricqhf.streamlit.app/

---

## 🚀 Features

### 📄 1. AI Resume Analyzer
- ATS-style semantic scoring
- NLP-based skill similarity
- Section-wise scoring breakdown
- Resume preview

### 🧠 2. Career Recommendation System
- Sentence Transformers (all-MiniLM-L6-v2)
- XGBoost multi-class classifier
- Top 3 career predictions with probability
- Interactive visualization using Altair

### 📊 3. Personalized Skill Roadmap
- Beginner → Intermediate → Advanced levels
- Dataset-driven skill progression
- Timeline-based structure

---

## 🧠 Tech Stack

- Python
- Streamlit
- Sentence Transformers
- XGBoost
- Scikit-learn
- Pandas / NumPy
- Altair
- PyPDF2

---

## Dataset

- Custom career-skill dataset
- Columns:
  - career
  - skills
  - experience_level
- Used for multi-class classification

---

## 📂 Project Structure
ai-career-guidance-system/
│
├── app.py
├── career_skills_dataset_levelwise_skills.csv
├── requirements.txt
└── README.md

---

## ⚙️ Machine Learning Pipeline

1. Text preprocessing
2. Sentence embeddings generation
3. Label encoding of careers
4. Train-test split
5. XGBoost multi-class classification
6. Probability-based top career ranking

---

## 📈 Model Performance

- Embedding Model: all-MiniLM-L6-v2
- Classifier: XGBoost
- Multi-class classification
- High accuracy on structured career dataset

---

## How to Run Locally
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-career-guidance-system.git
cd ai-career-guidance-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

   ---
   
## 🌍 SDG Impact

Supports **UN SDG 8 – Decent Work & Economic Growth**  
Helping students identify career paths through AI-driven insights.

---

## 🔥 Why This Project Stands Out

- Real-world applied NLP
- ML deployment experience
- End-to-end AI pipeline
- Internship-ready project
- Cloud deployment (Streamlit)

---

