import streamlit as st
import pickle
import requests
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")


st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
div[data-baseweb="select"] > div {
    background-color: #262730;
    color: white;
}
button {
    background-color: #ff4b4b !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD MODEL FILES ----------------
movies = pickle.load(open("movies_list.pkl", "rb"))
similarity = pickle.load(open("similarity_reduced.pkl","rb"))





@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=23d0eb31f08099e584b23f27aa9fd6cf&language=en-US"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return "https://via.placeholder.com/300x450?text=No+Image"

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/300x450?text=No+Image"

    except Exception as e:
        print("Poster fetch error:", e)
        return "https://via.placeholder.com/300x450?text=No+Image"


# ---------------- RECOMMENDER ----------------
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except:
        return ["Movie not found"], ["No poster"]

    rec_movies = []
    rec_posters = []

    for idx, score in similarity[movie_index]:
        movie_id = movies.iloc[idx]['movie_id']
        rec_movies.append(str(movies.iloc[idx]['title']))
        rec_posters.append(fetch_poster(movie_id))

    return rec_movies, rec_posters



# ---------------- UI ----------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommender System (ML Based)</h1>",
    unsafe_allow_html=True
)

st.write("")
movie_list = movies["title"].values
selected_movie = st.selectbox("🎥 Select a movie", movie_list)

st.write("")

if st.button("✨ Show Recommendations"):
    with st.spinner("Fetching recommendations..."):
        names, posters = recommend(str(selected_movie))



    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i], width=150)
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>{names[i]}</p>",
                unsafe_allow_html=True
            )
