from sklearn.metrics.pairwise import cosine_similarity
import re                 # to remove punctuation and special characters
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
import nltk               # For NLP tasks like stopwords
from nltk.corpus import stopwords
import pandas as pd # Import pandas to handle CSV files
nltk.download('stopwords')


# Load the dataset
movies = pd.read_csv('data/movies.csv')  # '..' because 'data' is one folder up from src

# Show the first 5 rows to check
print(movies.head())
# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # lowercase all text
        text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation & special characters
        words = text.split()  # split text into words
        words = [w for w in words if w not in stopwords.words('english')]  # remove stopwords
        return ' '.join(words)  # join words back into a single string
    else:
        return ''
movies['clean_overview'] = movies['overview'].apply(clean_text)
print(movies[['title', 'clean_overview']].head())
# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the cleaned text
tfidf_matrix = vectorizer.fit_transform(movies['clean_overview'])

# Check the shape of the matrix
print("TF-IDF matrix shape:", tfidf_matrix.shape)
# Compute cosine similarity between all movies
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Similarity matrix shape:", similarity.shape)
def recommend(movie_title):
    index = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(index) == 0:
        print("Movie not found! Please check the spelling.")
        return
    index = index[0]

    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = sorted_scores[1:6]  # top 5 similar

    print(f"\nMovies similar to '{movie_title}':")
    for i, score in top_movies:
        print(f"  {movies.iloc[i]['title']} (score: {score:.2f})")
while True:
    movie_name = input("\nEnter a movie name (or type 'exit' to quit): ")
    if movie_name.lower() == 'exit':
        break
    recommend(movie_name)
