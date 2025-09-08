import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests

# CONFIGURATION
TMDB_API_KEY = 'b228433a599248879f82263f980d561e'


# DATA LOADING & PROCESSING
@st.cache_data
@st.cache_data
def load_and_process_data():
    # Load data
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    # Merge datasets
    credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
    movies = movies_df.merge(credits_df, on='id')

    # Preserve title before doing anything else
    movies['title'] = movies_df['title']  # <--- FIX HERE

    # Functions to process fields
    def convert(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except:
            return []

    def get_director(obj):
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return i['name']
        except:
            return ''
        return ''

    def collapse(x):
        if isinstance(x, list):
            return " ".join(x)
        return x

    # Process fields
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].fillna('')
    movies['crew'] = movies['crew'].fillna('')

    # Create tags
    movies['tags'] = (
            movies['overview'] + ' ' +
            movies['genres'].apply(collapse) + ' ' +
            movies['keywords'].apply(collapse) + ' ' +
            movies['cast'].apply(collapse) + ' ' +
            movies['crew']
    )

    # Create final dataset
    final = movies[['title', 'tags']].copy()
    final['tags'] = final['tags'].apply(lambda x: x.lower())

    return final


movies = load_and_process_data()


# VECTORIZATION & SIMILARITY

@st.cache_resource
def compute_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['tags'])
    return cosine_similarity(tfidf_matrix)


similarity = compute_similarity_matrix(movies)


# TMDB POSTER FETCH
def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code != 200:
        return ""
    data = response.json()
    if data['results']:
        poster_path = data['results'][0].get('poster_path', None)
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return ""


# RECOMMENDATION FUNCTION

def fetch_poster_and_trailer(movie_title):
    global poster_url
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code != 200:
        return "", ""

    data = response.json()
    if data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0].get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

        # Now fetch trailer
        trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        trailer_resp = requests.get(trailer_url).json()

        for vid in trailer_resp.get("results", []):
            if vid["site"] == "YouTube" and vid["type"] == "Trailer":
                return poster_url, f"https://www.youtube.com/watch?v={vid['key']}"

    return poster_url, ""


def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return []

    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster_url, trailer_url = fetch_poster_and_trailer(title)
        recommended_movies.append((title, poster_url, trailer_url))

    return recommended_movies


# --- Page Setup ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# --- Header Section ---
st.markdown("""
<div style='
    text-align: center; 
    padding: 50px; 
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    border-radius: 15px; 
    color: white;
'>
    <h1 style='font-size: 48px; margin-bottom: 5px;'>🎬 Movie Recommender</h1>
    <p style='font-size: 18px;'>Get personalized movie suggestions based on your favorites!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Movie Selection ---
movie_list = movies['title'].dropna().str.strip()
movie_list = movie_list[~movie_list.str.startswith("#")]
movie_list = movie_list.sort_values().unique()

st.markdown("<h3 style='text-align: center; color: #ff7e5f;'>📽️ Select a Movie</h3>", unsafe_allow_html=True)
selected_movie = st.selectbox("", movie_list, index=0, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

# --- Recommendation Button Centered ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎯 Get Recommendations"):
        placeholder = st.empty()
        with placeholder.container():
            with st.spinner("Fetching recommendations... 🎬"):
                recommendations = recommend(selected_movie)

            if recommendations:
                # Recommendation Section with Gradient Background
                st.markdown("""
                <div style='
                    padding: 30px; 
                    border-radius: 15px; 
                    background: linear-gradient(135deg, #ffe6e1, #fff5f0);
                '>
                """, unsafe_allow_html=True)

                st.markdown("<h3 style='color: #ff7e5f;'>🎉 Recommended Movies</h3>", unsafe_allow_html=True)
                st.markdown("<hr style='border:1px solid #ff7e5f;'>", unsafe_allow_html=True)

                # Cards layout
                num_cols = min(len(recommendations), 5)
                cols = st.columns(num_cols, gap="medium")

                for col, (title, poster_url, trailer_url) in zip(cols, recommendations):
                    with col:
                        # Card container
                        st.markdown(
                            f"""
                            <div style='
                                background: linear-gradient(135deg, #ffffff, #ffe6e1);
                                padding: 10px;
                                border-radius: 12px;
                                text-align: center;
                                box-shadow: 0 6px 18px rgba(0,0,0,0.12);
                            '>
                            """, unsafe_allow_html=True
                        )

                        # Poster
                        st.image(poster_url or "https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)

                        # Title
                        st.markdown(f"<p style='font-weight: bold; font-size: 16px; margin-top: 8px;'>{title}</p>", unsafe_allow_html=True)

                        # Trailer Button
                        if trailer_url:
                            if st.button(f"🎥 Watch Trailer", key=title):
                                st.markdown(f"[Click here to watch trailer]({trailer_url})", unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another movie.")