
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st # Import streamlit here for the cache decorator

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True) # quiet=True to avoid printing to console on every run

# -------------------------------
# Caching the expensive setup steps
# -------------------------------
@st.cache_data
def setup_recommendation_system():
    # 1️⃣ Load the dataset
    # IMPORTANT: Make sure 'data/movies.csv' exists relative to where you run the app!
    try:
        movies = pd.read_csv('data/movies.csv')
    except FileNotFoundError:
        st.error("Movie dataset file not found. Make sure 'data/movies.csv' is in the correct location.")
        return None, None, None # Return None to indicate failure

    # 2️⃣ Clean text (lowercase, remove punctuation & stopwords)
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            words = text.split()
            # Use set for faster lookups
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
            return ' '.join(words)
        return ''

    # Apply cleaning - only if 'overview' column exists
    if 'overview' in movies.columns:
        movies['clean_overview'] = movies['overview'].apply(clean_text)
    else:
        st.error("The 'movies.csv' file must contain an 'overview' column.")
        return None, None, None

    # 3️⃣ Convert text to numeric vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    # Handle potential NaNs/Empty strings from cleaning before fitting
    tfidf_matrix = vectorizer.fit_transform(movies['clean_overview'].fillna(''))
    
    # 4️⃣ Compute cosine similarity between all movies
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Return all necessary computed data
    return movies, similarity_matrix

# Run the setup function once and store the results
movies_df, similarity_matrix = setup_recommendation_system()


# Define function to get similar movies
def get_recommendations(movie_title):
    # Check if setup was successful
    if movies_df is None or similarity_matrix is None:
        return []

    # find the movie index (use the cached df)
    index = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index
    if len(index) == 0:
        return []  # movie not found
    
    index = index[0]

    # get similarity scores for that movie
    scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = sorted_scores[1:6]  # skip itself, take top 5

    # return only the movie names
    return [movies_df.iloc[i]['title'] for i, _ in top_movies] 