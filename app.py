
import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10101a, #191b2b);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.movie-card {
    background: #25283b;
    padding: 16px;
    border-radius: 14px;
    min-height: 125px;
}
div.stButton > button {
    background: #e63946;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineMatch")
st.write("### Discover your next favorite movie")
st.caption("AI-powered recommendations based on genres and story descriptions.")

API_KEY = st.secrets.get("TMDB_API_KEY", "")

@st.cache_data(show_spinner=False)
def load_movies():
    return pd.read_csv("movies.csv")

@st.cache_data(ttl=3600, show_spinner=False)
def get_tmdb_details(title, api_key):
    if not api_key:
        return None, None, None

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": title},
            timeout=8
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            movie = results[0]
            poster_path = movie.get("poster_path")
            poster_url = (
                "https://image.tmdb.org/t/p/w500" + poster_path
                if poster_path else None
            )
            rating = movie.get("vote_average")
            overview = movie.get("overview")
            return poster_url, rating, overview

    except requests.RequestException:
        pass

    return None, None, None

movies = load_movies()
movies.columns = movies.columns.str.strip().str.lower()

required_columns = {"title", "genre", "description"}
if not required_columns.issubset(movies.columns):
    st.error("movies.csv must contain title, genre and description columns.")
    st.stop()

movies = movies.dropna(subset=["title"]).reset_index(drop=True)
movies["title"] = movies["title"].astype(str)
movies["genre"] = movies["genre"].fillna("").astype(str)
movies["description"] = movies["description"].fillna("").astype(str)
movies["features"] = movies["genre"] + " " + movies["description"]

if len(movies) < 2:
    st.error("Add at least two movies to movies.csv.")
    st.stop()

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies["features"])
similarity = cosine_similarity(feature_matrix)

st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox(
        "🎥 Choose a movie you like",
        movies["title"].tolist()
    )

with col2:
    number = st.selectbox(
        "Recommendations",
        [3, 5]
    )

if st.button("✨ Find My Movies", type="primary", use_container_width=True):
    movie_index = movies.index[
        movies["title"] == selected_movie
    ][0]

    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = [
        item for item in scores if item[0] != movie_index
    ][:number]

    st.subheader(f"Because you liked {selected_movie}")

    selected_poster, selected_rating, _ = get_tmdb_details(
        selected_movie, API_KEY
    )

    if selected_poster:
        st.image(selected_poster, width=160)

    if selected_rating is not None:
        st.caption(f"⭐ TMDB rating: {selected_rating:.1f}/10")

    for rank, (index, score) in enumerate(recommendations, start=1):
        movie = movies.iloc[index]
        poster, rating, overview = get_tmdb_details(
            movie["title"], API_KEY
        )

        with st.container(border=True):
            image_col, info_col = st.columns([1, 3])

            with image_col:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("### 🎞️")
                    st.caption("Poster unavailable")

            with info_col:
                st.markdown(f"### {rank}. {movie['title']}")

                if rating is not None:
                    st.write(f"⭐ **TMDB Rating:** {rating:.1f}/10")

                st.write(f"**Genre:** {movie['genre']}")
                st.write(
                    f"**Description:** {overview or movie['description']}"
                )
                st.caption(f"AI similarity score: {score:.0%}")

    if not API_KEY:
        st.info("Add your TMDB API key in .streamlit/secrets.toml to enable posters and ratings.")
    else:
        st.caption("Movie posters and ratings are provided by TMDB.")

st.divider()
st.caption("Built with Python • Streamlit • Pandas • Scikit-learn • TMDB")