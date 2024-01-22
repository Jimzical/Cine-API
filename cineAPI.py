from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from movie_functions import get_all_movies, get_movie_by_title, search_movie_title, get_content_based_recommendations,load_model
from movie_functions import get_popular_movies as popular_movies
from fastapi import Query


# Reading data
df = pd.read_csv('data/df_v3.csv')


cosine_similarity_matrix = load_model('model/content_based_model_v2.pkl.gz')

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

origins = [
    "http://localhost:3000",  # React app address
    # add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def get_movies() -> dict:
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
    return get_all_movies(df)

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
    return get_movie_by_title(movie_title, df)

@app.get("/recommendations/{movie}", tags=["Recommendations"])
async def get_movie_recommendations(
    movie: str ,
    limit: int = Query(5, ge=1, le=10, description="The number of recommendations to return")
) -> dict:
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


@app.get("/popular", tags=["Recommendations"])
async def get_popular_movies(
    sortby: str = Query('score', description="The metric to sort by", enum=['score', 'title', 'release_year']),
    limit: int = Query(10, ge=1, description="The number of movies to return")
) -> dict:
    """
    **Description:**
    Retrieves a list of popular movies sorted by a given metric.

    **Parameters:**
    - sortby (str, optional): The metric to sort by. Valid values are `score`, `title`, and `release_year`. Defaults to `score`.
    - limit (int, optional): The number of movies to return. Defaults to 10.

    **Returns:**
    A list of movie titles sorted by the given metric.

    **Example:**

    `curl http://localhost:8000/popular?sortby=profit&limit=5`
    **Example Response:**
    ```json
    {
    "movies": 
        [
            "Movie 1",
            "Movie 2",
            "Movie 3",
            "Movie 4",
            "Movie 5"
        ]
    }
    ```
    """
    return popular_movies(df,sortby=sortby, limit=limit)
    # return await get_popular_movies_func(sortby, limit)