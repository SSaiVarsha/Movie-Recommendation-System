import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")
movies["tags"] = movies["genre"] + " " + movies["keywords"]

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["tags"])
similarity = cosine_similarity(matrix)

pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))
print("Model saved!")
