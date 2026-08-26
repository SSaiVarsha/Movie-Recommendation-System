import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")
movies["tags"] = movies["genre"] + " " + movies["keywords"]

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["tags"])
similarity = cosine_similarity(matrix)

def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    print("\nRecommended Movies:\n")
    for i in scores:
        print(movies.iloc[i[0]].title)

name = input("Enter movie name: ")
recommend(name)
