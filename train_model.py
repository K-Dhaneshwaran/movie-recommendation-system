import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movie_recommendation_dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("movie_recommendation_dataset/tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def fetch_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i['name'])
    return L

movies['cast'] = movies['cast'].apply(fetch_cast)

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].fillna('')

movies['tags'] = movies['overview'] + " " + \
                 movies['genres'].apply(lambda x:" ".join(x)) + " " + \
                 movies['keywords'].apply(lambda x:" ".join(x)) + " " + \
                 movies['cast'].apply(lambda x:" ".join(x)) + " " + \
                 movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(new_df, open("movies_list.pkl","wb"))
pickle.dump(similarity, open("similarity.pkl","wb"))

print("✅ Model files created!")
