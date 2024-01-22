import gzip
import pickle
import pandas as pd
from fastapi import HTTPException
from fuzzywuzzy import process, fuzz


def load_model(model_path):
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_all_movies(df):
    return{"movies": df['original_title'].values.tolist()}

def get_movie_by_title(movie_title, df):
    """
    Retrieve movie details by title.

    Parameters:
    movie_title (str): The title of the movie to retrieve.
    df (DataFrame): The DataFrame containing the movie data.

    Returns:
    A dictionary containing the movie details, or raises an HTTPException if the movie is not found.
    """
    movie_title = movie_title.lower()
    movie_title = search_movie_title(movie_title, df['original_title'])

    movie = df[df['original_title'] == movie_title]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"movie": movie.to_dict(orient='records')}

def search_movie_title(query, movie_titles):
    matches = process.extract(query, movie_titles, scorer=fuzz.ratio)
    matches = [match for match in matches if match[1] >= 60]
    if matches:
        return matches[0][0]
    else:
        return None

def get_content_based_recommendations(movie_title, model_data, df, limit=6):
    movie_title = movie_title.lower()
    cosine_similarities = model_data['cosine_similarities']
    matched_title = search_movie_title(movie_title, df['original_title'])
    if matched_title is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    movie_index = df[df['original_title'] == matched_title].index[0]
    similar_movies = list(enumerate(cosine_similarities[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    similar_movies = similar_movies[1:limit]
    recommended_movie_titles = [df['original_title'].iloc[i[0]] for i in similar_movies]
    response = {"recommendations": recommended_movie_titles}
    return response

def get_popular_movies(df, sortby: str, limit: int):
    if sortby == 'score':
        sortby = 'vote_average'
    if sortby == 'title':
        sortby = 'original_title'

    res = {
        "movies": df.sort_values(by=sortby, ascending=False)['original_title'].head(limit).values.tolist()[1:]
    }

    return res