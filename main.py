import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cos_similarity():
    data = pd.read_csv('final_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data, sim


def rcmd(movie):
    movie = movie.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape

    except:
        data, sim = cos_similarity()

    # check if the movie is in our database or not
    if movie not in data['movie_title'].unique():
        return 'Sorry! This movie is not in our database. Please check the spelling or try with some other movies'
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title'] == movie].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key=lambda x: x[1], reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containing all 10 movie recommendations
        recommended = []
        for i in range(len(lst)):
            a = lst[i][0]
            recommended.append(data['movie_title'][a])
        return recommended


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/recommend")
def recommend():
    # user input
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()

    if type(r) == type('string'):
        return render_template('recommend.html', movie=movie, r=r, t='s')
    else:
        return render_template('recommend.html', movie=movie, r=r, t='l')


if __name__ == '__main__':
    app.run(debug=True)
