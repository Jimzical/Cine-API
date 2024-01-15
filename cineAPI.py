from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from fuzzywuzzy import process, fuzz
import gzip

# Reading data
df = pd.read_csv('data/df_v3.csv')

# # Importing model
with gzip.open('model/content_based_model_v2.pkl.gz', 'rb') as f:
    cosine_similarity_matrix = pickle.load(f)
    

app = FastAPI(
    title="CineAPI",
    description="A simple API for movie recommendations",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Recommendations",
            "description": "Endpoints for getting movie recommendations"
        },
        {
            "name": "Movies",
            "description": "Endpoints for working with movie data"
        },
    ]
)

def search_movie_title(query, movie_titles):
    """
    Search for a movie title in the list of movie titles.

    **Parameters:**
    query (str): The query string to search for.
    movie_titles (list): The list of movie titles to search in.

    **Returns:**
    str: The best matching movie title, or None if no good match was found.
    """
    # Extract all matches without a score cutoff
    matches = process.extract(query, movie_titles, scorer=fuzz.ratio)

    # Filter matches based on score cutoff
    matches = [match for match in matches if match[1] >= 60]

    # Return the best match or None
    if matches:
        return matches[0][0]  # Return only the match string
    else:
        return None

def get_content_based_recommendations(movie_title : str, model_data : dict, df : pd.DataFrame, limit : int = 6):
    """
    Get content-based recommendations for a movie.

    **Parameters:**
    movie_title (str): The title of the movie to get recommendations for.
    model_data (dict): The model data (cosine similarity matrix).
    df (DataFrame): The dataframe containing the movie data.
    limit (int): The number of recommendations to return.[ Default: 5 ]

    **Returns:**
    dict: The list of top 5 recommended movies.

    **Example:**
    >>> get_content_based_recommendations('Inception', cosine_similarity_matrix, df)
    {'recommendations': ['Inception', 'The Matrix', 'The Matrix Revolutions', 'The Matrix Reloaded', 'The Terminator']}
    """

 
    # making movie title to be lowercase
    movie_title = movie_title.lower()

    cosine_similarities = model_data['cosine_similarities']

    # Use fuzzy matching to find the closest match to the input movie title
    matched_title = search_movie_title(movie_title, df['original_title'])

    if matched_title is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Find the index of the given movie title in the dataframe
    movie_index = df[df['original_title'] == matched_title].index[0]
    # Calculate cosine similarities with other movies
    similar_movies = list(enumerate(cosine_similarities[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    # Get the top 5 recommendations
    similar_movies = similar_movies[1:limit]  # Skip the first movie because it's the input movie itself

    # Map the movie indices back to movie titles
    recommended_movie_titles = [df['original_title'].iloc[i[0]] for i in similar_movies]

    # Convert the list of movie names to a dictionary
    response = {"recommendations": recommended_movie_titles}

    return response

@app.get("/", tags=["General"])
async def root() -> dict:
    """
    **Description:**
    Root endpoint of the API.

    **Returns:**
    A welcome message.

    **Example:**

    `curl http://localhost:8000/`
    """
    return {
        "message": "Welcome to CineAPI!"
        }

@app.get("/movies", tags=["Movies"])
async def get_all_movies() -> dict:
    """
    **Description:**
    Retrieves a list of all available movies.

    **Returns:**
    A list of movie titles.

    **Example:**

    `curl http://localhost:8000/movies`
    
    **Example Response:**
    ```json
    {"movies": [
        "The Matrix", 
        "Inception", 
        "The Dark Knight", 
        "Interstellar", 
        "The Lord of the Rings: The Fellowship of the Ring"
        ]
    }
    ```
    """
    return {"movies": df['original_title'].values.tolist()}

@app.get("/movies/{movie_title}", tags=["Movies"])

async def get_movie_details(movie_title: str) -> dict:
    """
    **Description:**
    Retrieves details of a specific movie.

    **Parameters:**
    - movie_title (str): The title of the movie to retrieve details for.

    Returns:
    A dictionary containing the movie's details, or an HTTP 404 error if not found.

    **Example:**

    `curl http://localhost:8000/movies/The%20Matrix`

    **Example Response:**
    ```json
    {
    "movie": [
        {
        "id": 603,
        "popularity": 7.753899,
        "original_title": "the matrix",
        "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss Hugo Weaving Gloria Foster",
        "director": "Lilly Wachowski|Lana Wachowski",
        "keywords": "saving the world artificial intelligence man vs machine philosophy prophecy",
        "runtime": 136,
        "genres": "Action Science Fiction",
        "release_date": "3/30/1999",
        "vote_count": 6351,
        "vote_average": 7.8,
        "release_year": 1999,
        "budget_adj": 82470329.34,
        "revenue_adj": 606768749.7,
        "profit": 524298420.36,
        "release_month": 3
        }
    ]
    }
    ```
    """
    movie_title = movie_title.lower()
    movie_title = search_movie_title(movie_title, df['original_title'])

    movie = df[df['original_title'] == movie_title]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"movie": movie.to_dict(orient='records')}

@app.get("/recommendations/{movie}", tags=["Recommendations"])
async def get_movie_recommendations(movie: str, limit: int = 5) -> dict:
    """
    **Description:**
    Retrieves top N recommendations for a specific movie.

    **Parameters:**
    - movie (str): The title of the movie to get recommendations for.
    - limit (int, optional): The number of recommendations to return. Defaults to 5.

    **Returns:**
    A list of the top N recommended movie titles.

    **Example:**

    `curl http://localhost:8000/recommendations/Inception?limit=10`

    **Example Response:**
    ```json
    {
    "recommendations": 
        [
            "Inception", 
            "The Matrix",
            "The Matrix Revolutions", 
            "The Matrix Reloaded", 
            "The Terminator",
            "Movie 6",
            "Movie 7",
            "Movie 8",
            "Movie 9",
            "Movie 10"
        ]
    }
    ```
    """
    return get_content_based_recommendations(movie, cosine_similarity_matrix, df, limit)