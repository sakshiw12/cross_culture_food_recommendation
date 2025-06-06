import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df = model['data']
tfidf = model['tfidf']
tfidf_matrix = model['tfidf_matrix']

# Set Streamlit page config
st.set_page_config(page_title="Cross-Cultural_Food_Recommendation", layout="centered")

# Styling and title
st.markdown(
    """
    <style>
    .main {
        background-color: #f6f9fc;
        font-family: Arial;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='text-align: center;'>🌍 Cross-Cultural Food Recommendation System 🍜</h2>", unsafe_allow_html=True)

# Dropdown for dish selection
dish_name = st.selectbox("Select a dish to get recommendations:", df['name'].tolist())

# Recommend button
if st.button("Recommend Similar Dishes"):
    if dish_name not in df['name'].values:
        st.error("❌ Dish not found. Please select a valid dish.")
    else:
        idx = df[df['name'] == dish_name].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:6]
        recommendations = df.iloc[[i[0] for i in sim_scores]][['name', 'cuisine', 'diet', 'ingredients']]
        
        st.markdown("### ✅ Recommended Dishes:")
        st.dataframe(recommendations.reset_index(drop=True))




