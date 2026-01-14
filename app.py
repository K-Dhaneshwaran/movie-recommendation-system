import streamlit as st
import pickle
import requests
from dotenv import load_dotenv
import os

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
similarity = pickle.load(open("similarity.pkl", "rb"))





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
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    rec_movies = []
    rec_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]]['movie_id']
        rec_movies.append(movies.iloc[i[0]].title)
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
        names, posters = recommend(selected_movie)

    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i], width=150)
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>{names[i]}</p>",
                unsafe_allow_html=True
            )
