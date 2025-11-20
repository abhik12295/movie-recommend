# import pandas as pd
# import pickle
# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# data = pickle.load(open("movies.pkl", "rb"))
# sim = pickle.load(open("similarity.pkl", "rb"))
# print("Loaded successfully!")

# # def cos_similarity():
# #     data = pd.read_csv('final_data.csv')
# #     # creating a count matrix
# #     cv = CountVectorizer()
# #     count_matrix = cv.fit_transform(data['comb'])
# #     # creating a similarity score matrix
# #     sim = cosine_similarity(count_matrix)
# #     return data, sim


# # def rcmd(movie):
# #     movie = movie.lower()

# #     data, sim = cos_similarity()

# #     # check if the movie is in our database or not
# #     if movie not in data['movie_title'].unique():
# #         return 'Sorry! This movie is not in our database. Please check the spelling or try with some other movies'
# #     else:
# #         # getting the index of the movie in the dataframe
# #         i = data.loc[data['movie_title'] == movie].index[0]

# #         # fetching the row containing similarity scores of the movie
# #         # from similarity matrix and enumerate it
# #         lst = list(enumerate(sim[i]))

# #         # sorting this list in decreasing order based on the similarity score
# #         lst = sorted(lst, key=lambda x: x[1], reverse=True)

# #         # taking top 1- movie scores
# #         # not taking the first index since it is the same movie
# #         lst = lst[1:11]

# #         # making an empty list that will containing all 10 movie recommendations
# #         recommended = []
# #         for i in range(len(lst)):
# #             a = lst[i][0]
# #             recommended.append(data['movie_title'][a])
# #         return recommended


# app = Flask(__name__)


# def rcmd(movie):
#     movie = movie.lower()

#     # check if the movie is in our database
#     if movie not in data['movie_title'].unique():
#         return 'Sorry! This movie is not in our database. Please check the spelling or try with some other movies'
    
#     # index of the movie
#     i = data.loc[data['movie_title'] == movie].index[0]

#     # similarity row
#     lst = list(enumerate(sim[i]))

#     # sort
#     lst = sorted(lst, key=lambda x: x[1], reverse=True)[1:11]

#     # get recommendations
#     recommended = [data['movie_title'][a] for a, _ in lst]

#     return recommended



# @app.route("/")
# def home():
#     return render_template('home.html')


# @app.route("/recommend")
# def recommend():
#     # user input
#     movie = request.args.get('movie')
#     r = rcmd(movie)
#     movie = movie.upper()

#     if type(r) == type('string'):
#         return render_template('recommend.html', movie=movie, r=r, t='s')
#     else:
#         return render_template('recommend.html', movie=movie, r=r, t='l')


# # if __name__ == '__main__':
# #     app.run(debug=True)



import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

# Load all model data ONCE at startup
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend")
def recommend():
    movie = request.args.get("movie")

    if not movie:
        return render_template("recommend.html", movie="", r="No movie given", t="s")

    movie_lower = movie.lower()

    if movie_lower not in data['movie_title'].str.lower().values:
        return render_template("recommend.html", movie=movie.upper(),
                               r="Movie not found!", t="s")

    # index of movie
    idx = data.index[data['movie_title'].str.lower() == movie_lower][0]

    # compute similarity ON DEMAND (only vector vs all)
    movie_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()

    # get top 10 excluding itself
    top_indices = sim_scores.argsort()[::-1][1:11]

    recommendations = data['movie_title'].iloc[top_indices].tolist()

    return render_template("recommend.html",
                           movie=movie.upper(),
                           r=recommendations,
                           t="l")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=100)
